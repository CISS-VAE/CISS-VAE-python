"""
Optuna-based hyperparameter tuning for CISS-VAE.
This module defines:
- :class:`SearchSpace`: a structured container describing tunable/fixed hyperparameters.
- :func:`autotune`: runs Optuna trials that train CISSVAE models and selects the best trial
  by validation MSE, then retrains a final model with the best settings.
"""
import torch
import optuna
import json
import pandas as pd
from torch.utils.data import DataLoader
from ciss_vae.classes.vae import CISSVAE
from ciss_vae.classes.cluster_dataset import ClusterDataset
from ciss_vae.training.train_initial import train_vae_initial
from ciss_vae.training.train_refit import impute_and_refit_loop
from ciss_vae.utils.helpers import compute_val_mse
from itertools import combinations, product
import random
import sys

class SearchSpace:
    r"""
        Defines tunable and fixed hyperparameter ranges for the Optuna search.
        Parameters are specified as:
        * **scalar**: fixed value (e.g., ``latent_dim=16``)
        * **list**: categorical choice (e.g., ``hidden_dims=[64, 128, 256]``)
        * **tuple**: range (``(min, max)``) for ``suggest_int``/``suggest_float``
        Attributes
        ----------
        num_hidden_layers : int | list[int] | tuple[int, int]
            Number of encoder/decoder hidden layers.
        hidden_dims : int | list[int] | tuple[int, int]
            Hidden dimension spec. If int → repeated per layer; list → per‑layer choices; tuple → range.
        latent_dim : int | tuple[int, int]
            Latent size or range.
        latent_shared : bool | list[bool]
            Whether latent is shared (or categorical choice).
        output_shared : bool | list[bool]
            Whether output is shared (or categorical choice).
        lr : float | tuple[float, float]
            Initial learning rate or range.
        decay_factor : float | tuple[float, float]
            LR exponential decay factor or range.
        beta : float | tuple[float, float]
            KL weight or range.
        num_epochs : int | tuple[int, int]
            Epochs for initial training (or range).
        batch_size : int | tuple[int, int]
            Mini-batch size (or range).
        num_shared_encode : list[int]
            Candidate counts of shared encoder layers.
        num_shared_decode : list[int]
            Candidate counts of shared decoder layers.
        refit_patience : int | tuple[int, int]
            Early-stop patience for refit loops (or range).
        refit_loops : int | tuple[int, int]
            Maximum number of refit loops (or range).
        epochs_per_loop : int | tuple[int, int]
            Epochs per refit loop (or range).
        reset_lr_refit : bool | list[bool]
            Whether to reset LR before refit.
        """
    def __init__(self,
                 num_hidden_layers=(1, 4),
                 hidden_dims=[64, 512],
                 latent_dim=[10, 100],
                 latent_shared=[True, False],
                 output_shared=[True,False],
                 lr=(1e-4, 1e-3),
                 decay_factor=(0.9, 0.999),
                 beta=0.01,
                 num_epochs=1000,
                 batch_size=64,
                 num_shared_encode=[0, 1, 3],
                 num_shared_decode=[0, 1, 3],
                 refit_patience=2,
                 refit_loops=100,
                 epochs_per_loop = 1000,
                 reset_lr_refit = [True, False]):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.latent_shared = latent_shared
        self.output_shared = output_shared
        self.lr = lr
        self.decay_factor = decay_factor
        self.beta = beta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_shared_encode = num_shared_encode
        self.num_shared_decode = num_shared_decode
        self.refit_patience = refit_patience
        self.refit_loops = refit_loops
        self.epochs_per_loop = epochs_per_loop
        self.reset_lr_refit = reset_lr_refit

def autotune(
    search_space: SearchSpace,
    train_dataset: ClusterDataset,
    save_model_path=None,
    save_search_space_path=None,
    n_trials=20,
    study_name="vae_autotune",
    device_preference="cuda",
    optuna_dashboard_db=None,
    load_if_exists=True,
    seed = 42,
    verbose = False,
    permute_hidden_layers: bool = True,
    constant_layer_size: bool = False,
    evaluate_all_orders: bool = False,
    max_exhaustive_orders: int = 100,
):
    r"""
    Optuna-based hyperparameter search for the CISSVAE model.
    Runs initial training, then impute–refit loops per trial, and selects the
    trial with the lowest validation MSE.
    :param search_space: Hyperparameter ranges and fixed values.
    :type search_space: SearchSpace
    :param train_dataset: Dataset containing masks, normalization, and cluster labels.
    :type train_dataset: ciss_vae.classes.cluster_dataset.ClusterDataset
    :param save_model_path: Optional path to save the best model's ``state_dict``.
    :type save_model_path: str | None
    :param save_search_space_path: Optional path to dump the resolved search-space.
    :type save_search_space_path: str | None
    :param n_trials: Number of Optuna trials to run.
    :type n_trials: int
    :param study_name: Name for the Optuna study.
    :type study_name: str
    :param device_preference: ``"cuda"`` or ``"cpu"``; falls back to CPU if CUDA unavailable.
    :type device_preference: str
    :param optuna_dashboard_db: RDB storage URL/file for dashboard (or ``None`` for in‑memory).
    :type optuna_dashboard_db: str | None
    :param load_if_exists: Load existing study with the same name from storage.
    :type load_if_exists: bool
    :param seed: Base RNG seed for order generation etc.
    :type seed: int
    :param verbose: If ``True``, prints diagnostic logs.
    :type verbose: bool
    :param permute_hidden_layers: If ``True``, randomize layer orders per trial.
    :type permute_hidden_layers: bool
    :param constant_layer_size: If ``True``, all hidden layers use the same size.
    :type constant_layer_size: bool
    :returns: Tuple ``(best_imputed_df, best_model, study, results_df)``
    :rtype: (pandas.DataFrame, CISSVAE, optuna.study.Study, pandas.DataFrame)
    :raises ValueError: If search space parameters are malformed or incompatible.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # --------------------------
    # Infer device
    # --------------------------
    if device_preference == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_preference)
    direction="minimize"
    
    # --------------------------
    # Infer input dim and num clusters
    # --------------------------
    input_dim = train_dataset.raw_data.shape[1]
    num_clusters = len(torch.unique(train_dataset.cluster_labels))
    
    # --------------------------
    # Helper to sample from fixed, categorical, or range param
    # --------------------------
    def sample_param(trial, name, value):
        if isinstance(value, (int, float, bool, str)):
            return value
        elif isinstance(value, list):
            return trial.suggest_categorical(name, value)
        elif isinstance(value, tuple):
            if all(isinstance(v, int) for v in value):
                return trial.suggest_int(name, value[0], value[1])
            elif all(isinstance(v, float) for v in value):
                return trial.suggest_float(name, value[0], value[1], log=value[0] > 0)
        raise ValueError(f"Unsupported parameter format for '{name}': {value}")
    
    # --------------------------
    # Helpers to format order of shared/unshared + control for enumerating or sampling form orders
    # --------------------------
    def _format_order(order_list):
        """['shared','unshared',...] → 'S,U,...' (stable, readable, categorical)"""
        abbrev = {'shared': 'S', 'unshared': 'U'}
        return ",".join(abbrev[x] for x in order_list)
    
    def _decode_pattern(p: str):
        """'S,U,S' → ['shared','unshared','shared']"""
        m = {'S': 'shared', 'U': 'unshared'}
        return [m[x] for x in str(p).split(",")]
    
    def _enumerate_orders(n_layers: int, n_shared: int):
        """Deterministically enumerate **all** valid orders (no randomness)."""
        if n_layers < 0 or n_shared < 0 or n_shared > n_layers:
            return []
        patterns = []
        for idxs in combinations(range(n_layers), n_shared):
            arr = ['U'] * n_layers
            for i in idxs:
                arr[i] = 'S'
            patterns.append(",".join(arr))
        return patterns
    
    def _canonical_orders(n_layers: int, nse: int, nsd: int):
        """Canonical (non-permuted) layout"""
        enc_list = (["unshared"] * (n_layers - nse)) + (["shared"] * nse)
        dec_list = (["shared"] * nsd) + (["unshared"] * (n_layers - nsd))
        return _format_order(enc_list), _format_order(dec_list)
    
    def _build_order(style: str, n_layers: int, n_shared: int, rng: random.Random):
        n_shared = max(0, min(int(n_shared), int(n_layers)))
        shared_positions = list(range(n_layers))
        if style == "shared_tail":
            pos = list(range(n_layers - n_shared, n_layers))
        elif style == "shared_head":
            pos = list(range(0, n_shared))
        elif style == "alternating":
            pos = list(range(0, n_layers, max(1, n_layers // max(1, n_shared))))[:n_shared] if n_shared > 0 else []
        elif style == "random":
            pos = rng.sample(shared_positions, n_shared)
        else:
            pos = list(range(n_layers - n_shared, n_layers))  # fallback
        arr = ['unshared'] * n_layers
        for i in pos:
            arr[i] = 'shared'
        return arr
    
    # --------------------------
    # Create Optuna objective
    # --------------------------
    def objective(trial):
        if verbose:
            print(f"\nStarting Trial {trial.number + 1}/{n_trials}")
        
        # --------------------------
        # Parse Parameters
        # --------------------------
        num_hidden_layers = sample_param(trial, "num_hidden_layers", search_space.num_hidden_layers)
        
        # ---- Hidden dimensions ----
        if constant_layer_size:
            base_dim = sample_param(trial, "hidden_dim_constant", search_space.hidden_dims)
            hidden_dims = [base_dim] * num_hidden_layers
        else:
            hidden_dims = [
                sample_param(trial, f"hidden_dim_{i}", search_space.hidden_dims)
                for i in range(num_hidden_layers)
            ]
        
        latent_dim = sample_param(trial, "latent_dim", search_space.latent_dim)
        latent_shared = sample_param(trial, "latent_shared", search_space.latent_shared)
        output_shared = sample_param(trial, "output_shared", search_space.output_shared)
        learning_rate = sample_param(trial, "lr", search_space.lr)
        decay_factor = sample_param(trial, "decay_factor", search_space.decay_factor)
        beta = sample_param(trial, "beta", search_space.beta)
        num_epochs = sample_param(trial, "num_epochs", search_space.num_epochs)
        batch_size = sample_param(trial, "batch_size", search_space.batch_size)
        
        # Handle num_shared_encode/decode 
        nse_raw = trial.suggest_categorical("num_shared_encode", sorted(set(search_space.num_shared_encode)))
        nsd_raw = trial.suggest_categorical("num_shared_decode", sorted(set(search_space.num_shared_decode)))
        num_shared_encode = int(min(int(nse_raw), int(num_hidden_layers)))
        num_shared_decode = int(min(int(nsd_raw), int(num_hidden_layers)))
        order_style_enc = trial.suggest_categorical("order_style_enc", ["shared_tail", "shared_head", "alternating", "random"])
        order_style_dec = trial.suggest_categorical("order_style_dec", ["shared_head", "shared_tail", "alternating", "random"])
        order_seed_enc = trial.suggest_int("order_seed_enc", 0, 10_000)
        order_seed_dec = trial.suggest_int("order_seed_dec", 0, 10_000)
        
        refit_patience = sample_param(trial, "refit_patience", search_space.refit_patience)
        refit_loops = sample_param(trial, "refit_loops", search_space.refit_loops)
        epochs_per_loop = sample_param(trial, "epochs_per_loop", search_space.epochs_per_loop)
        reset_lr_refit = sample_param(trial, "reset_lr_refit", search_space.reset_lr_refit)
        trial.set_user_attr("num_shared_encode_effective", int(num_shared_encode))
        trial.set_user_attr("num_shared_decode_effective", int(num_shared_decode))
        lr_refit = learning_rate if reset_lr_refit else None

        if verbose:
            print(f"  Parameters: layers={num_hidden_layers}, latent_dim={latent_dim}, lr={learning_rate:.2e}")

        # --------------------------
        # Build orders
        # --------------------------
        if evaluate_all_orders:
            enc_pool = _enumerate_orders(num_hidden_layers, num_shared_encode)
            dec_pool = _enumerate_orders(num_hidden_layers, num_shared_decode)
            combos_to_eval = list(product(enc_pool, dec_pool))
        else:
            if permute_hidden_layers:
                rng_enc = random.Random(seed * 9973 + order_seed_enc)
                rng_dec = random.Random(seed * 9967 + order_seed_dec)
                layer_order_enc = _build_order(order_style_enc, num_hidden_layers, num_shared_encode, rng_enc)
                layer_order_dec = _build_order(order_style_dec, num_hidden_layers, num_shared_decode, rng_dec)
            else:
                enc_pat, dec_pat = _canonical_orders(num_hidden_layers, num_shared_encode, num_shared_decode)
                layer_order_enc = _decode_pattern(enc_pat)
                layer_order_dec = _decode_pattern(dec_pat)
            combos_to_eval = [(_format_order(layer_order_enc), _format_order(layer_order_dec))]

        best_val = None
        best_patterns = None
        best_refit_history_df = None

        # --------------------------
        # Train each combination
        # --------------------------
        for enc_pat, dec_pat in combos_to_eval:
            layer_order_enc = _decode_pattern(enc_pat)
            layer_order_dec = _decode_pattern(dec_pat)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            model = CISSVAE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                layer_order_enc=layer_order_enc,
                layer_order_dec=layer_order_dec,
                latent_shared=latent_shared,
                num_clusters=num_clusters,
                latent_dim=latent_dim,
                output_shared=output_shared
            ).to(device)

            if verbose:
                print(f"  Starting initial training...")

            # Initial training
            model = train_vae_initial(
                model=model,
                train_loader=train_loader,
                epochs=num_epochs,
                initial_lr=learning_rate,
                decay_factor=decay_factor,
                beta=beta,
                device=device,
                verbose=verbose
            )
            
            if verbose:
                print(f"  Starting refit loops...")

            # Impute && refit loop
            _, model, _, refit_history_df = impute_and_refit_loop(
                model=model,
                train_loader=train_loader,
                max_loops=refit_loops,
                patience=refit_patience,
                epochs_per_loop=epochs_per_loop,
                initial_lr=lr_refit,
                decay_factor=decay_factor,
                beta=beta,
                device=device,
                verbose=verbose,
                batch_size=batch_size,
            )

            # Get validation MSE
            val_mse = compute_val_mse(model, train_loader.dataset, device)
            if (best_val is None) or (val_mse < best_val):
                best_val = val_mse
                best_patterns = (enc_pat, dec_pat)
                best_refit_history_df = refit_history_df

        if verbose:
            print(f"  Trial {trial.number + 1} complete - Validation MSE: {best_val:.6f}")

        # Report intermediate values to Optuna
        if best_refit_history_df is not None and "val_mse" in best_refit_history_df.columns:
            for i, v in enumerate(best_refit_history_df["val_mse"]):
                if pd.notna(v):
                    trial.report(float(v), step=i)

        # Record the chosen best patterns for this trial
        if best_patterns is not None:
            trial.set_user_attr("best_layer_order_enc", best_patterns[0])
            trial.set_user_attr("best_layer_order_dec", best_patterns[1])

        return best_val

    # -----------------------
    # Optuna study setup
    # -----------------------
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=optuna_dashboard_db,
        load_if_exists=load_if_exists
    )

    # -----------------------
    # Run optimization
    # -----------------------
    print(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    print(f"Optimization complete. Best trial: {study.best_trial.number} (MSE: {study.best_value:.6f})")

    # -----------------------
    # Final model training
    # -----------------------
    print("Training final model with best parameters...")

    best_params = study.best_trial.params 
    
    def get_best_param(name):
        if name in best_params:
            return best_params[name]
        else:
            return getattr(search_space, name)

    # Get best parameters
    best_num_hidden_layers = get_best_param("num_hidden_layers")
    
    if constant_layer_size:
        if "hidden_dim_constant" in best_params:
            base_dim = best_params["hidden_dim_constant"]
        else:
            base_dim = getattr(search_space, "hidden_dims", 64)
            if isinstance(base_dim, (list, tuple)):
                base_dim = base_dim[0]
        best_hidden_dims = [int(base_dim)] * best_num_hidden_layers
    else:
        if f"hidden_dim_0" in best_params:
            best_hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_num_hidden_layers)]
        else:
            hdims = get_best_param("hidden_dims")
            if isinstance(hdims, list):
                if len(hdims) == 1:
                    best_hidden_dims = hdims * best_num_hidden_layers
                elif len(hdims) < best_num_hidden_layers:
                    best_hidden_dims = (hdims * best_num_hidden_layers)[:best_num_hidden_layers]
                else:
                    best_hidden_dims = hdims[:best_num_hidden_layers]
            else:
                best_hidden_dims = [hdims] * best_num_hidden_layers

    # Get best layer orders
    ua = study.best_trial.user_attrs
    nse_eff = int(ua.get("num_shared_encode_effective",
                         min(int(best_params.get("num_shared_encode", 0)), int(best_num_hidden_layers))))
    nsd_eff = int(ua.get("num_shared_decode_effective",
                         min(int(best_params.get("num_shared_decode", 0)), int(best_num_hidden_layers))))
    enc_pat = study.best_trial.user_attrs.get("best_layer_order_enc")
    dec_pat = study.best_trial.user_attrs.get("best_layer_order_dec")
    if enc_pat is None or dec_pat is None:
        enc_pat, dec_pat = _canonical_orders(best_num_hidden_layers, nse_eff, nsd_eff)

    best_layer_order_enc = _decode_pattern(enc_pat)
    best_layer_order_dec = _decode_pattern(dec_pat)
    latent_shared = bool(get_best_param("latent_shared"))
    output_shared = bool(get_best_param("output_shared"))
    latent_dim = int(get_best_param("latent_dim"))
    num_epochs = int(get_best_param("num_epochs"))
    lr = float(get_best_param("lr"))
    decay_factor = float(get_best_param("decay_factor"))
    beta = float(get_best_param("beta"))
    batch_size = int(get_best_param("batch_size"))

    # ---------------------------
    # Build && train final model
    # ---------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    best_model = CISSVAE(
        input_dim=input_dim,
        hidden_dims=best_hidden_dims,
        layer_order_enc=best_layer_order_enc,
        layer_order_dec=best_layer_order_dec,
        latent_shared=latent_shared,
        num_clusters=num_clusters,
        latent_dim=latent_dim,
        output_shared=output_shared
    ).to(device)

    # Final training
    best_model = train_vae_initial(
        model=best_model,
        train_loader=train_loader,
        epochs=num_epochs,
        initial_lr=lr,
        decay_factor=decay_factor,
        beta=beta,
        device=device,
        verbose=verbose
    )

    best_imputed_df, best_model, _, _ = impute_and_refit_loop(
        model=best_model,
        train_loader=train_loader,
        max_loops=search_space.refit_loops,
        patience=search_space.refit_patience,
        epochs_per_loop=search_space.epochs_per_loop,
        initial_lr=lr,
        decay_factor=decay_factor,
        beta=beta,
        device=device,
        verbose=verbose
    )

    print("Final model training complete.")

    # -----------------------
    # Save results
    # -----------------------
    if save_model_path:
        torch.save(best_model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    if save_search_space_path:
        with open(save_search_space_path, "w") as f:
            json.dump(search_space.__dict__, f, indent=4)
        print(f"Search space saved to {save_search_space_path}")

    # Create results DataFrame
    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "val_mse": t.value, **t.params}
        row["layer_order_enc_used"] = (
            t.params.get("layer_order_enc")
            or t.user_attrs.get("best_layer_order_enc")
            or t.user_attrs.get("layer_order_enc_used")
        )
        row["layer_order_dec_used"] = (
            t.params.get("layer_order_dec")
            or t.user_attrs.get("best_layer_order_dec")
            or t.user_attrs.get("layer_order_dec_used")
        )
        rows.append(row)
    results_df = pd.DataFrame(rows)

    return best_imputed_df, best_model, study, results_df