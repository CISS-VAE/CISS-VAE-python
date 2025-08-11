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
from tqdm import tqdm
import random

class TQDMProgress:
    """Custom callback to show progress bar for Optuna trials."""
    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials, desc="Optuna tuning", leave=True)
    def __call__(self, study, trial):
        self.pbar.update(1)
    def close(self):
        self.pbar.close()

class SearchSpace:
    """Defines tunable and fixed hyperparameter ranges. Input for 'autotune' function.
    [] -> pick items from list.
    () -> tuple, creates a range (min, max, step).
     x -> use single value"""
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
    train_dataset: ClusterDataset,                   # ClusterDataset object
    save_model_path=None,
    save_search_space_path=None,
    n_trials=20,
    study_name="vae_autotune",
    device_preference="cuda",
    show_progress=False,
    optuna_dashboard_db=None,
    load_if_exists=True,
    seed = 42,
    verbose = False,
    permute_hidden_layers: bool = True,   # default: expose layer order as Optuna categorical params
    constant_layer_size: bool = False,    # if True, one dim is tuned and repeated for all layers
    evaluate_all_orders: bool = False,    # if True, try ALL valid enc/dec orders per trial and pick best
    max_exhaustive_orders: int = 100,    # safety cap for evaluate_all_orders
):
    """
    Optuna-based hyperparameter search for VAE model.
        Parameters:
            - search_space : SearchSpace
                The hyperparameter search space for the tuning
            - train_dataset : ClusterDataset
                A ClusterDataset object                 
            - save_model_path : 
                [optional], Where to save the best model
            - save_search_space_path : 
                [optional], Where to save the search space
            - n_trials : int 
                Number of trials to run
            - study_name : str default="vae_autotune"
                Name for the optuna study, default is "vae_autotune",
            - device_preference : (str), 
                Preferred device, default is "cuda",
            - show_progress : (bool) default=False, 
                Set as True to show progress bar, default is False,
            - optuna_dashboard_db : (str), 
                Path to optuna dashboard db file. Default is None,
            - load_if_exists : (bool) default=True, 
                If study by 'study_name' exists in optuna dashboard db, load that study. Default is True.
            - seed : (int): 
                Seed for selecting order of shared/unshared layers. Default is 42.
            - verbose : (bool) default=False: 
                Set to True to print diagnostic statements. Default is False
            permute_hidden_layers : (bool) default=TRUE:
                *True*: Encoder/decoder shared/unshared **order** is represented as **categorical
                Optuna parameters** (`layer_order_enc`, `layer_order_dec`) chosen from all valid
                permutations given the number of shared layers. This makes order visible in
                parameter-importance plots on the Optuna dashboard.

                *False*: Use a canonical, deterministic layout (no search over order):
                - ENCODER: shared layers at the **END** (unshared...unshared, shared...shared)
                - DECODER: shared layers at the **START** (shared...shared, unshared...unshared)
            - constant_layer_size : (bool) default=False:
                If True, Optuna samples a single `hidden_dim_constant` and we **repeat** it
                across all hidden layers (cleaner importances, fewer degrees of freedom).
                Otherwise, tune one `hidden_dim_i` per layer.
            - evaluate_all_orders : (bool) default=False:
                If True, each trial **exhaustively evaluates *all* valid order pairs**
                (given `num_hidden_layers`, `num_shared_encode`, `num_shared_decode`)
                and returns the best metric among them. In this mode, layer order is
                *not* an Optuna parameter (so it won't appear in importance), but the
                chosen best orders are recorded in `trial.user_attrs` and included in
                `results_df`. Guarded by `max_exhaustive_orders` to avoid explosion.
            - max_exhaustive_orders : (int) default=100:
                If `evaluate_all_orders`, sets safety cap for number of runs to try. Can be increased/decreased as needed. 


        Returns: 
            best_imputed_df : pd.DataFrame
                The imputed dataset from the best model
            best_model : CISSVAE 
                The best model
            study : 
                Optuna study
            results_df : pd.DataFrame
                The results of the autotuning
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
        """
        'S,U,S' → ['shared','unshared','shared']
        """
        m = {'S': 'shared', 'U': 'unshared'}
        return [m[x] for x in str(p).split(",")]

    def _enumerate_orders(n_layers: int, n_shared: int):
        """
        Deterministically enumerate **all** valid orders (no randomness).
        Each order is a string like 'S,U,U,S'. Uses combinations to place S.
        """
        if n_layers < 0 or n_shared < 0 or n_shared > n_layers:
            return []
        patterns = []
        for idxs in combinations(range(n_layers), n_shared):
            arr = ['U'] * n_layers
            for i in idxs:
                arr[i] = 'S'
            patterns.append(",".join(arr))  # lexicographic by construction
        return patterns

    def _canonical_orders(n_layers: int, nse: int, nsd: int):
        """
        Canonical (non-permuted) layout:
          - ENCODER: unshared ... unshared, shared ... shared   (shared at END)
          - DECODER: shared ... shared, unshared ... unshared   (shared at START)
        Returns patterns as 'S,U,...' strings.
        """
        enc_list = (["unshared"] * (n_layers - nse)) + (["shared"] * nse)
        dec_list = (["shared"] * nsd) + (["unshared"] * (n_layers - nsd))
        return _format_order(enc_list), _format_order(dec_list)


    # --------------------------
    # Create Optuna objective
    # --------------------------
    def objective(trial):
        # --------------------------
        # Parse Parameters
        # --------------------------
        num_hidden_layers = sample_param(trial, "num_hidden_layers", search_space.num_hidden_layers)
        # ---- Hidden dimensions ----
        if constant_layer_size:
            # One parameter drives all hidden dims (shows up in importance)
            base_dim = sample_param(trial, "hidden_dim_constant", search_space.hidden_dims)
            hidden_dims = [base_dim] * num_hidden_layers
        else:
            # One param per layer
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
        
        ## handle num_shared_encode/decode 
        enc_choices = [v for v in search_space.num_shared_encode if v <= num_hidden_layers]
        enc_choices = enc_choices or [num_hidden_layers]
        num_shared_encode = trial.suggest_categorical("num_shared_encode", enc_choices)
        # num_shared_encode = sample_param(trial, "num_shared_encode", search_space.num_shared_encode)
        dec_choices = [v for v in search_space.num_shared_decode if v <= num_hidden_layers]
        dec_choices = dec_choices or [num_hidden_layers]
        num_shared_decode = trial.suggest_categorical("num_shared_decode", dec_choices)
        # num_shared_decode = sample_param(trial, "num_shared_decode", search_space.num_shared_decode)
        refit_patience = sample_param(trial, "refit_patience", search_space.refit_patience)
        refit_loops = sample_param(trial, "refit_loops", search_space.refit_loops)
        epochs_per_loop = sample_param(trial, "epochs_per_loop", search_space.epochs_per_loop)
        reset_lr_refit = sample_param(trial, "reset_lr_refit", search_space.reset_lr_refit)

        if reset_lr_refit:
            lr_refit = learning_rate
        else:
            lr_refit = None

        # --------------------------
        # Handle hidden layers
        # Allow for permutation of hidden layers
        # Create shared/unshared layer orders for encoder & decoder
        # layer_order_enc: List of str: "shared" or "unshared"
        # layer_order_dec: List of str: "shared" or "unshared"
        # --------------------------

        enc_pool = _enumerate_orders(num_hidden_layers, num_shared_encode)  # e.g., ['S,U,U','U,S,U',...]
        dec_pool = _enumerate_orders(num_hidden_layers, num_shared_decode)

        # Determine which orders to evaluate this trial
        if permute_hidden_layers and not evaluate_all_orders:
            # Expose order as **categorical parameters** (visible in Optuna importance)
            enc_pattern = trial.suggest_categorical("layer_order_enc", enc_pool)
            dec_pattern = trial.suggest_categorical("layer_order_dec", dec_pool)
            combos_to_eval = [(enc_pattern, dec_pattern)]

            # Record for convenience
            trial.set_user_attr("layer_order_enc_used", enc_pattern)
            trial.set_user_attr("layer_order_dec_used", dec_pattern)

        elif evaluate_all_orders:
            # Exhaustively evaluate enc×dec order pairs up to the cap; if more, warn & sample.
            total = len(enc_pool) * len(dec_pool)
            if total <= max_exhaustive_orders:
                combos_to_eval = list(product(enc_pool, dec_pool))
            else:
                warnings.warn(
                    f"Exhaustive order eval would run {total} models; sampling "
                    f"{max_exhaustive_orders} combos uniformly without replacement.",
                    RuntimeWarning
                )
                # Reproducible uniform sampling without building the full product list.
                # Index the Cartesian product by i ∈ [0, total):
                #   enc_idx = i // len(dec_pool), dec_idx = i % len(dec_pool)
                rng = random.Random(seed + trial.number)
                k = max_exhaustive_orders
                # Guard: if k > total (e.g., user sets a huge cap), clamp to total
                k = min(k, total)
                sampled_indices = rng.sample(range(total), k=k)
                combos_to_eval = [
                    (enc_pool[i // len(dec_pool)], dec_pool[i % len(dec_pool)])
                    for i in sampled_indices
                ]

        else:
            # Non-permuted **canonical** layout:
            #   ENCODER shared at END, DECODER shared at START
            enc_pattern, dec_pattern = _canonical_orders(num_hidden_layers, num_shared_encode, num_shared_decode)
            combos_to_eval = [(enc_pattern, dec_pattern)]
            trial.set_user_attr("layer_order_enc_used", enc_pattern)
            trial.set_user_attr("layer_order_dec_used", dec_pattern)


        best_val = None
        best_patterns = None
        best_val_mse_history = None ## lets us plot the val_mse_history

        for enc_pat, dec_pat in combos_to_eval:
            layer_order_enc = _decode_pattern(enc_pat)
            layer_order_dec = _decode_pattern(dec_pat)

            # --------------------------
            # Train loader and model
            # --------------------------
            
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

            # --------------------------
            # Initial training
            # --------------------------
            model = train_vae_initial(
                model=model,
                train_loader=train_loader,
                epochs=num_epochs,
                initial_lr=learning_rate,
                decay_factor=decay_factor,
                beta=beta,
                device=device,
                verbose=False
            )

            # --------------------------
            # Impute & refit loop
            # --------------------------
            _, model, _, val_mse_history = impute_and_refit_loop(
                model=model,
                train_loader=train_loader,
                max_loops=refit_loops,
                patience=refit_patience,
                epochs_per_loop=epochs_per_loop,
                initial_lr=lr_refit,
                decay_factor=decay_factor,
                beta=beta,
                device=device,
                verbose=False,
                batch_size=batch_size
            )

            # --------------------------
            # Get validation MSE from the training-refit loop
            # --------------------------
            val_mse = compute_val_mse(model, train_loader.dataset, device)

            if (best_val is None) or (val_mse < best_val):
                best_val = val_mse
                best_patterns = (enc_pat, dec_pat)
                best_val_mse_history = val_mse_history

        # Report intermediate values to Optuna (so dashboard shows the learning curve)
        
        for i, val_mse in enumerate(best_val_mse_history):
            trial.report(val_mse, step=i)
        
        # Record the chosen best patterns for this trial (useful in exhaustive/sample mode)
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
    # Progress bar
    # -----------------------
    if show_progress:
        progress = TQDMProgress(n_trials=n_trials)
        study.optimize(objective, n_trials=n_trials, callbacks=[progress])
        progress.close()
    else:
        study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params

    # -----------------------
    # Final model: train with best params from scratch (robust to fixed params)
    # -----------------------

    # Helper to get best param or fallback to search_space default
    def get_best_param(name):
        if name in best_params:
            return best_params[name]
        else:
            return getattr(search_space, name)

    # 1. num_hidden_layers (int)
    best_num_hidden_layers = get_best_param("num_hidden_layers")

    # 2. hidden_dims (list of int)
    # If hidden_dims were tuned, best_params will have keys like "hidden_dim_0", else just use fixed or list from search_space
    if constant_layer_size:
        # If we tuned it, it will be present; otherwise fall back to search_space
        if "hidden_dim_constant" in best_params:
            base_dim = best_params["hidden_dim_constant"]
        else:
            base_dim = getattr(search_space, "hidden_dims", 64)  # conservative fallback
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



    # 3. Layer orders (reproducible from Optuna trial, no shuffle)
    # Retrieve best encoder/decoder patterns for the best trial
    def _get_best_patterns_from_study(study_: optuna.Study):
        p = study_.best_trial.params
        ua = study_.best_trial.user_attrs
        # If permuted (non-exhaustive), patterns live in params; else in user_attrs.
        enc = p.get("layer_order_enc") or ua.get("best_layer_order_enc") or ua.get("layer_order_enc_used")
        dec = p.get("layer_order_dec") or ua.get("best_layer_order_dec") or ua.get("layer_order_dec_used")
        return enc, dec

    enc_pattern, dec_pattern = _get_best_patterns_from_study(study)

    # If missing (very unlikely), fall back to canonical using best counts
    if enc_pattern is None or dec_pattern is None:
        best_nse = int(get_best_param("num_shared_encode"))
        best_nsd = int(get_best_param("num_shared_decode"))
        enc_pattern, dec_pattern = _canonical_orders(best_num_hidden_layers, best_nse, best_nsd)

    best_layer_order_enc = _decode_pattern(enc_pattern)
    best_layer_order_dec = _decode_pattern(dec_pattern)

    latent_shared = bool(get_best_param("latent_shared"))
    output_shared = bool(get_best_param("output_shared"))
    latent_dim = int(get_best_param("latent_dim"))
    num_epochs = int(get_best_param("num_epochs"))
    lr = float(get_best_param("lr"))
    decay_factor = float(get_best_param("decay_factor"))
    beta = float(get_best_param("beta"))
    batch_size = int(get_best_param("batch_size"))

    # ---------------------
    # Build & train the final best model
    # ---------------------

    # Prepare DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create best model
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

    # Train initial fit
    best_model = train_vae_initial(
        model=best_model,
        train_loader=train_loader,
        epochs=num_epochs,
        initial_lr=lr,
        decay_factor=decay_factor,
        beta=beta,
        device=device,
        verbose=False
    )

    # Impute & refit
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
        verbose=False
    )

    # -----------------------
    # Save things if needed
    # -----------------------
    if save_model_path:
        torch.save(best_model.state_dict(), save_model_path)

    if save_search_space_path:
        with open(save_search_space_path, "w") as f:
            json.dump(search_space.__dict__, f, indent=4)

    # Results DataFrame (include layer orders per trial for inspection)
    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "val_mse": t.value, **t.params}
        # Include which order actually got used/evaluated-best even if not a param
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
