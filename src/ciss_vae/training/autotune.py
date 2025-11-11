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
from pathlib import Path
# NEW: Rich imports for track() function
from rich.progress import track
from rich.console import Console

class SearchSpace:
    """Defines tunable and fixed hyperparameter ranges for the Optuna search.
    
    Parameters are specified as:  
    - **scalar**: fixed value (e.g., ``latent_dim=16``)  
    - **list**: categorical choice (e.g., ``hidden_dims=[64, 128, 256]``)  
    - **tuple**: range ``(min, max)`` for ``suggest_int`` or ``suggest_float``  
    
    :param num_hidden_layers: Number of encoder/decoder hidden layers, defaults to (1, 4)
    :type num_hidden_layers: int or list[int] or tuple[int, int], optional
    :param hidden_dims: Hidden dimension specification - int for repeated per layer, list for per-layer choices, tuple for range, defaults to [64, 512]
    :type hidden_dims: int or list[int] or tuple[int, int], optional
    :param latent_dim: Latent dimension size or range, defaults to [10, 100]
    :type latent_dim: int or tuple[int, int], optional
    :param latent_shared: Whether latent space is shared across clusters, defaults to [True, False]
    :type latent_shared: bool or list[bool], optional
    :param output_shared: Whether output layer is shared across clusters, defaults to [True, False]
    :type output_shared: bool or list[bool], optional
    :param lr: Initial learning rate or range, defaults to (1e-4, 1e-3)
    :type lr: float or tuple[float, float], optional
    :param decay_factor: Learning rate exponential decay factor or range, defaults to (0.9, 0.999)
    :type decay_factor: float or tuple[float, float], optional
    :param beta: KL divergence weight or range, defaults to 0.01
    :type beta: float or tuple[float, float], optional
    :param num_epochs: Number of epochs for initial training, defaults to 1000
    :type num_epochs: int or tuple[int, int], optional
    :param batch_size: Mini-batch size, defaults to 64
    :type batch_size: int or tuple[int, int], optional
    :param num_shared_encode: Candidate counts of shared encoder layers, defaults to [0, 1, 3]
    :type num_shared_encode: list[int], optional
    :param num_shared_decode: Candidate counts of shared decoder layers, defaults to [0, 1, 3]
    :type num_shared_decode: list[int], optional
    :param encoder_shared_placement: Strategy for arranging shared vs unshared layers in encoder, defaults to ["at_end", "at_start", "alternating", "random"]
    :type encoder_shared_placement: list[str], optional
    :param decoder_shared_placement: Strategy for arranging shared vs unshared layers in decoder, defaults to ["at_end", "at_start", "alternating", "random"]
    :type decoder_shared_placement: list[str], optional
    :param refit_patience: Early-stop patience for refit loops, defaults to 2
    :type refit_patience: int or tuple[int, int], optional
    :param refit_loops: Maximum number of refit loops, defaults to 100
    :type refit_loops: int or tuple[int, int], optional
    :param epochs_per_loop: Number of epochs per refit loop, defaults to 1000
    :type epochs_per_loop: int or tuple[int, int], optional
    :param reset_lr_refit: Whether to reset learning rate before refit, defaults to [True, False]
    :type reset_lr_refit: bool or list[bool], optional
    """
    def __init__(self,
                 num_hidden_layers=(1, 4),
                 hidden_dims=[64, 512],
                 latent_dim=[10, 100],
                 latent_shared=[True, False],
                 output_shared=[True,False],
                 lr=(1e-4, 1e-3),
                 decay_factor=(0.9, 0.999),
                 weight_decay =0.001,
                 beta=0.01,
                 num_epochs=1000,
                 batch_size=64,
                 num_shared_encode=[0, 1, 3],
                 num_shared_decode=[0, 1, 3],
                 encoder_shared_placement=["at_end", "at_start", "alternating", "random"],
                 decoder_shared_placement=["at_end", "at_start", "alternating", "random"],
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
        self.weight_decay = weight_decay
        self.beta = beta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_shared_encode = num_shared_encode
        self.num_shared_decode = num_shared_decode
        self.encoder_shared_placement = encoder_shared_placement
        self.decoder_shared_placement = decoder_shared_placement
        self.refit_patience = refit_patience
        self.refit_loops = refit_loops
        self.epochs_per_loop = epochs_per_loop
        self.reset_lr_refit = reset_lr_refit
    
    def _as_jsonable(self):
        """Return a dict of fields with tuples converted to lists (JSON-safe)."""
        def convert(x):
            if isinstance(x, tuple):
                return [convert(v) for v in x]
            if isinstance(x, list):
                return [convert(v) for v in x]
            if isinstance(x, dict):
                return {k: convert(v) for k, v in x.items()}
            return x
        return {k: convert(v) for k, v in self.__dict__.items()}

    def save(self, file_path):
        """Save this search space to a JSON file.
        :param file_path: Path to save file.
        :type file_path: string
        """
        p = Path(file_path)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self._as_jsonable(), f, indent=2)

    @classmethod
    def load(cls, file_path):
        """Load a search space from a JSON file and return a new instance.
        :param file_path: Path to saved SearchSpace.
        :type file_path: string
        """
        p = Path(file_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Note: JSON has lists, not tuples. The constructor accepts lists just fine.
        return cls(**data)

    def __str__(self):
        """Readable summary showing which parameters are tunable vs fixed."""
        lines = ["SearchSpace("]
        for k, v in self.__dict__.items():
            tunable = isinstance(v, (list, tuple))
            flag = "TUNABLE" if tunable else "FIXED"
            lines.append(f"  {k}: {v!r}  [{flag}]")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self):
        """Compact representation useful for debugging."""
        tunables = [k for k, v in self.__dict__.items() if isinstance(v, (list, tuple))]
        fixed = [k for k in self.__dict__ if k not in tunables]
        return (
            f"<SearchSpace tunable={tunables} fixed={fixed}>"
            .replace("tunable", str(tunables))
            .replace("fixed", str(fixed))
        )

    


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
    show_progress = False,  # NEW: Added back progress parameter
    # permute_hidden_layers: bool = True,
    constant_layer_size: bool = False,
    evaluate_all_orders: bool = False,
    max_exhaustive_orders: int = 100,
    return_history: bool = False,
    n_jobs = 1, ## add param to docs,
    debug = False,

):
    r"""Optuna-based hyperparameter search for the CISSVAE model.
    
    Runs initial training followed by impute-refit loops per trial, selecting the
    trial with the lowest validation MSE. The best model is then retrained with
    optimal hyperparameters and returned along with the imputed dataset.
    
    :param search_space: Hyperparameter ranges and fixed values for optimization
    :type search_space: SearchSpace
    :param train_dataset: Dataset containing masks, normalization, and cluster labels
    :type train_dataset: ClusterDataset
    :param save_model_path: Optional path to save the best model's state_dict, defaults to None
    :type save_model_path: str, optional
    :param save_search_space_path: Optional path to dump the resolved search-space configuration, defaults to None
    :type save_search_space_path: str, optional
    :param n_trials: Number of Optuna trials to run, defaults to 20
    :type n_trials: int, optional
    :param study_name: Name identifier for the Optuna study, defaults to "vae_autotune"
    :type study_name: str, optional
    :param device_preference: Preferred device ("cuda" or "cpu"), falls back to CPU if CUDA unavailable, defaults to "cuda"
    :type device_preference: str, optional
    :param optuna_dashboard_db: RDB storage URL/file for Optuna dashboard or None for in-memory, defaults to None
    :type optuna_dashboard_db: str, optional
    :param load_if_exists: Whether to load existing study with the same name from storage, defaults to True
    :type load_if_exists: bool, optional
    :param seed: Base random number generator seed for reproducible order generation, defaults to 42
    :type seed: int, optional
    :param verbose: Whether to print detailed diagnostic logs during training, defaults to False
    :type verbose: bool, optional
    :param show_progress: Whether to display Rich progress bars during training, defaults to False
    :type show_progress: bool, optional
    :param constant_layer_size: Whether all hidden layers should use the same dimension size, defaults to False
    :type constant_layer_size: bool, optional
    :param evaluate_all_orders: Whether to permute and evaluate all possible shared/unshared layer orders, defaults to False
    :type evaluate_all_orders: bool, optional
    :param max_exhaustive_orders: Maximum number of layer order permutations to test when evaluate_all_orders is True, defaults to 100
    :type max_exhaustive_orders: int, optional
    :param return_history: Whether to return MSE training history dataframe of the best model, defaults to False
    :type return_history: bool, optional
    :param n_jobs: Number of jobs to run for autotuning (passed to optuna). Defaults to 1
    :type n_jobs: int, optional
    :param debug: Defaults to False. Set True for informative debugging statements.
    :type debug: bool, optional
    
    :return: Tuple containing (best_imputed_dataframe, best_model, optuna_study_object, results_dataframe, optional[best_model_history_df])
    :rtype: tuple[pandas.DataFrame, CISSVAE, optuna.study.Study, pandas.DataFrame] or tuple[pandas.DataFrame, CISSVAE, optuna.study.Study, pandas.DataFrame, pandas.DataFrame]
    
    :raises ValueError: If search space parameters are malformed or incompatible
    :raises RuntimeError: If CUDA is requested but not available and fallback fails
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # NEW: Initialize Rich console
    console = Console()
    
    # --------------------------
    # Infer device
    # --------------------------
    if device_preference == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_preference)
    direction="minimize"

    ## Save search space if asked
    if save_search_space_path is not None:
        search_space.save(save_search_space_path)
    
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
    
    ## don't need this since we have the things from build order
    # def _canonical_orders(n_layers: int, nse: int, nsd: int):
    #     """Canonical (non-permuted) layout. Unshared at beginning of encoder and end of decoder"""
    #     enc_list = (["unshared"] * (n_layers - nse)) + (["shared"] * nse)
    #     dec_list = (["shared"] * nsd) + (["unshared"] * (n_layers - nsd))
    #     return _format_order(enc_list), _format_order(dec_list)
    
    def _build_order(style: str, n_layers: int, n_shared: int, rng: random.Random):
        n_shared = max(0, min(int(n_shared), int(n_layers)))
        shared_positions = list(range(n_layers))
        if style == "at_end":
            pos = list(range(n_layers - n_shared, n_layers))
        elif style == "at_start":
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
    
    # ------------------------------------------------
    # Wrapper for train_initial and impute_refit functions that uses track()
    # ------------------------------------------------
    def train_vae_initial_with_progress(model, train_loader, epochs, initial_lr, decay_factor, weight_decay, beta, device, verbose_inner=False):
        """Wrapper for train_vae_initial that uses Rich track() for progress"""
        if show_progress:
            # Create a simple progress tracker using track()
            for epoch in track(range(epochs), description="Initial training"):
                # Call the actual training function for one epoch at a time
                # Note: This assumes train_vae_initial can be called with epochs=1
                model = train_vae_initial(
                    model=model,
                    train_loader=train_loader,
                    epochs=1,
                    initial_lr=initial_lr,
                    decay_factor=decay_factor,
                    beta=beta,
                    device=device,
                    weight_decay = weight_decay,
                    verbose=False  # Disable verbose to avoid spam
                )
        else:
            # Call original function without progress
            model = train_vae_initial(
                model=model,
                train_loader=train_loader,
                epochs=epochs,
                initial_lr=initial_lr,
                decay_factor=decay_factor,
                weight_decay = weight_decay,
                beta=beta,
                device=device,
                verbose=verbose_inner
            )
        return model
    
    def impute_and_refit_loop_with_progress(model, train_loader, max_loops, patience, epochs_per_loop, 
                                          initial_lr, decay_factor, weight_decay, beta, device, verbose_inner=False, batch_size=64, ):
        """Wrapper for impute_and_refit_loop that uses Rich track() for progress"""
        if show_progress:
            # Estimate total epochs for progress bar
            estimated_total_epochs = max_loops * epochs_per_loop
            
            # Use track() to show progress over estimated epochs
            epoch_counter = 0
            progress_iter = track(range(estimated_total_epochs), description="Refit loops")
            progress_iter = iter(progress_iter)  # Convert to iterator
            
            # Create a callback that advances the progress bar
            def progress_callback(n=1):
                nonlocal epoch_counter
                try:
                    for _ in range(n):
                        next(progress_iter)
                        epoch_counter += 1
                except StopIteration:
                    pass  # Progress bar completed
            
            # Call the original function with progress callback
            return impute_and_refit_loop(
                model=model,
                train_loader=train_loader,
                max_loops=max_loops,
                patience=patience,
                epochs_per_loop=epochs_per_loop,
                initial_lr=initial_lr,
                decay_factor=decay_factor,
                weight_decay = weight_decay,
                beta=beta,
                device=device,
                verbose=verbose_inner,
                batch_size=batch_size,
                progress_epoch=progress_callback
            )
        else:
            # Call original function without progress
            return impute_and_refit_loop(
                model=model,
                train_loader=train_loader,
                max_loops=max_loops,
                patience=patience,
                epochs_per_loop=epochs_per_loop,
                initial_lr=initial_lr,
                decay_factor=decay_factor,
                weight_decay = weight_decay,
                beta=beta,
                device=device,
                verbose=verbose_inner,
                batch_size=batch_size,
            )
    
    # --------------------------
    # Create Optuna objective
    # --------------------------
    def objective(trial):
        # NEW: Use Rich console for trial progress
        if show_progress:
            console.print(f"\n[green]Trial {trial.number}/{n_trials}")
        elif verbose:
            print(f"\nStarting Trial {trial.number}/{n_trials}")

        # ----------------
        # Check train_dataset for na causers
        # -----------------
        with torch.no_grad():
            cl = train_dataset.cluster_labels.clone()
            uniq = torch.unique(cl).cpu().tolist()
            remap = {old:i for i,old in enumerate(uniq)}
            new_cl = cl.cpu().apply_(lambda v: remap[int(v)]).to(train_dataset.cluster_labels.device)
        train_dataset.cluster_labels = new_cl

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
        weight_decay = sample_param(trial, "weight_decay", search_space.weight_decay)
        beta = sample_param(trial, "beta", search_space.beta)
        num_epochs = sample_param(trial, "num_epochs", search_space.num_epochs)
        batch_size = sample_param(trial, "batch_size", search_space.batch_size)
        
        # Handle num_shared_encode/decode ## updated 16SEP2025
        nse_raw = sample_param(trial, "num_shared_encode", search_space.num_shared_encode)
        nsd_raw = sample_param(trial, "num_shared_decode", search_space.num_shared_decode)
        num_shared_encode = max(0, min(int(nse_raw), int(num_hidden_layers)))
        num_shared_decode = max(0, min(int(nsd_raw), int(num_hidden_layers)))


        encoder_shared_placement = trial.suggest_categorical("encoder_shared_placement", search_space.encoder_shared_placement)
        decoder_shared_placement = trial.suggest_categorical("decoder_shared_placement", search_space.decoder_shared_placement)
        
        refit_patience = sample_param(trial, "refit_patience", search_space.refit_patience)
        refit_loops = sample_param(trial, "refit_loops", search_space.refit_loops)
        epochs_per_loop = sample_param(trial, "epochs_per_loop", search_space.epochs_per_loop)
        reset_lr_refit = sample_param(trial, "reset_lr_refit", search_space.reset_lr_refit)

        trial.set_user_attr("num_shared_encode_effective", int(num_shared_encode))
        trial.set_user_attr("num_shared_decode_effective", int(num_shared_decode))

        lr_refit = learning_rate if reset_lr_refit else None
        
        ## Show parameters with Rich (progress bar)
        if show_progress:
            console.print(f"Parameters: layers={num_hidden_layers}, latent_dim={latent_dim}, lr={learning_rate:.2e}, batch_size={batch_size}")
        elif verbose:
            print(f"  Parameters: layers={num_hidden_layers}, latent_dim={latent_dim}, lr={learning_rate:.2e}")
        
        # --------------------------
        # Build orders for shared/unshared layers
        # --------------------------
        if evaluate_all_orders:
            enc_pool = _enumerate_orders(num_hidden_layers, num_shared_encode)
            dec_pool = _enumerate_orders(num_hidden_layers, num_shared_decode)
            combos_to_eval = list(product(enc_pool, dec_pool))

            # FIXED: Apply max_exhaustive_orders limit !!
            if len(combos_to_eval) > max_exhaustive_orders:
                rng_ord = random.Random(seed * 9912 + trial.number)  # Use trial number for reproducibility
                combos_to_eval = rng_ord.sample(combos_to_eval, k=max_exhaustive_orders)

        else:
            # -----------------------
            # Reproducable seeds for random generated layer order
            # - uses _build_order() to determine the appropriate layer orders based on enc/decoder_shared_placement
            # -----------------------
            rng_enc = random.Random(seed * 9973 + trial.number)
            rng_dec = random.Random(seed * 9967 + trial.number)
            layer_order_enc = _build_order(encoder_shared_placement, num_hidden_layers, num_shared_encode, rng_enc)
            layer_order_dec = _build_order(decoder_shared_placement, num_hidden_layers, num_shared_decode, rng_dec)
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
                output_shared=output_shared,
                binary_feature_mask = train_dataset.binary_feature_mask,
                debug = debug
            ).to(device)
            
            ## Use progress wrappers instead of original functions
            model = train_vae_initial_with_progress(
                model=model,
                train_loader=train_loader,
                epochs=num_epochs,
                initial_lr=learning_rate,
                decay_factor=decay_factor,
                weight_decay=weight_decay,
                beta=beta,
                device=device,
                verbose_inner=verbose
            )
            
            _, model, _ = impute_and_refit_loop_with_progress(
                model=model,
                train_loader=train_loader,
                max_loops=refit_loops,
                patience=refit_patience,
                epochs_per_loop=epochs_per_loop,
                initial_lr=lr_refit,
                decay_factor=decay_factor,
                weight_decay=weight_decay,
                beta=beta,
                device=device,
                verbose_inner=verbose,
                batch_size=batch_size,
            )
            
            # Get validation MSE
            imputation_error, val_mse, val_bce = compute_val_mse(model, train_loader.dataset, device)
            if (best_val is None) or (imputation_error < best_val):
                best_val = imputation_error
                best_patterns = (enc_pat, dec_pat)
                best_refit_history_df = model.training_history_
                trial.set_user_attr("best_val_mse", val_mse)
                trial.set_user_attr("best_val_bce", val_bce)
        
        # Show completion with Rich
        if show_progress:
            console.print(f"✓ Trial {trial.number + 1} complete - Total Imputation Error: {best_val:.4f}")
        elif verbose:
            print(f"  Trial {trial.number + 1} complete - Total Imputation Error: {best_val:.6f}")
        
        # Report intermediate values to Optuna
        if best_refit_history_df is not None and "imputation_error" in best_refit_history_df.columns:
            for i, v in enumerate(best_refit_history_df["imputation_error"]):
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
    
    study.set_metric_names(["Total Imputation Error"]) 
    
    # -----------------------
    # Run optimization
    # -----------------------
    # Use Rich console for study start
    if show_progress:
        console.print(f"[bold blue]Starting Optuna optimization with {n_trials} trials...")
    else:
        print(f"Starting Optuna optimization with {n_trials} trials...")
    
    study.optimize(objective, n_trials=n_trials, n_jobs = n_jobs)
    
    # Use Rich console for completion
    if show_progress:
        console.print(f"\n[bold green]✓ Optimization complete!")
        console.print(f"Best trial: {study.best_trial.number} (Total Imputation Error: {study.best_value:.6f})")
    else:
        print(f"Optimization complete. Best trial: {study.best_trial.number} (Total Imputation Error: {study.best_value:.6f})")
    
    # -----------------------
    # Final model training
    # -----------------------
    if show_progress:
        console.print(f"\n[bold cyan]Training final model with best parameters...")
    else:
        print("Training final model with best parameters...")
    
    # ---------------------
    # Extract best params for final model training
    # ---------------------
    best_params = study.best_trial.params 
    
    def get_best_param(name):
        if name in best_params:
            return best_params[name]
        else:
            return getattr(search_space, name)
    
    ## get num hidden layers
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
    
    ## get layer orders
    ua = study.best_trial.user_attrs
    nse_eff = int(ua.get("num_shared_encode_effective",
                         min(int(best_params.get("num_shared_encode", 0)), int(best_num_hidden_layers))))
    nsd_eff = int(ua.get("num_shared_decode_effective",
                         min(int(best_params.get("num_shared_decode", 0)), int(best_num_hidden_layers))))
    enc_pat = study.best_trial.user_attrs.get("best_layer_order_enc")
    dec_pat = study.best_trial.user_attrs.get("best_layer_order_dec")
    
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
    refit_patience = int(get_best_param("refit_patience"))
    refit_loops = int(get_best_param("refit_loops"))
    epochs_per_loop = int(get_best_param("epochs_per_loop"))
    reset_lr_refit = bool(get_best_param("reset_lr_refit"))
    weight_decay = float(get_best_param("weight_decay"))
    # ---------------------------
    # Build & train final model
    # ---------------------------
    # ---------------------------
    # Build & train final model
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
        output_shared=output_shared,
        binary_feature_mask = train_dataset.binary_feature_mask,
        debug = debug
    ).to(device)
    
    # Initialize history tracking for final model
    final_model_history = None
    
    # Use track() for final training too
    if show_progress:
        # Final initial training with progress
        if return_history:
            initial_history_list = []
            for epoch in track(range(num_epochs), description="Final initial training"):
                result = train_vae_initial(
                    model=best_model,
                    train_loader=train_loader,
                    epochs=1,
                    initial_lr=lr,
                    decay_factor=decay_factor,
                    weight_decay = weight_decay,
                    beta=beta,
                    device=device,
                    verbose=False,
                    return_history=True
                )
                if isinstance(result, tuple):
                    best_model, epoch_history = result
                    initial_history_list.append(epoch_history)
                else:
                    best_model = result
        else:
            for epoch in track(range(num_epochs), description="Final initial training"):
                best_model = train_vae_initial(
                    model=best_model,
                    train_loader=train_loader,
                    epochs=1,
                    initial_lr=lr,
                    decay_factor=decay_factor,
                    weight_decay = weight_decay,
                    beta=beta,
                    device=device,
                    verbose=False,
                    return_history=False
                )
        
        # Final refit with progress
        estimated_refit_epochs = refit_loops * epochs_per_loop
        if isinstance(refit_loops, tuple):
            estimated_refit_epochs = refit_loops[0] * epochs_per_loop
        if isinstance(epochs_per_loop, tuple):
            estimated_refit_epochs = refit_loops * epochs_per_loop[0]
        
        progress_iter = track(range(estimated_refit_epochs), description="Final refit loops")
        progress_iter = iter(progress_iter)
        
        def final_progress_callback(n=1):
            try:
                for _ in range(n):
                    next(progress_iter)
            except StopIteration:
                pass
        
        best_imputed_df, best_model, _ = impute_and_refit_loop(
            model=best_model,
            train_loader=train_loader,
            max_loops=refit_loops,
            patience=refit_patience,
            epochs_per_loop=epochs_per_loop,
            initial_lr=lr,
            decay_factor=decay_factor,
            weight_decay = weight_decay,
            beta=beta,
            device=device,
            verbose=verbose,
            progress_epoch=final_progress_callback
        )
        
        # Combine initial and refit histories if requested
        if return_history:
            final_model_history = best_model.training_history_
    else:
        # Final training without progress
        if return_history:
            result = train_vae_initial(
                model=best_model,
                train_loader=train_loader,
                epochs=num_epochs,
                initial_lr=lr,
                decay_factor=decay_factor,
                weight_decay = weight_decay,
                beta=beta,
                device=device,
                verbose=verbose,
                return_history=True
            )
            if isinstance(result, tuple):
                best_model, initial_history_df = result
            else:
                best_model = result
                initial_history_df = None
        else:
            best_model = train_vae_initial(
                model=best_model,
                train_loader=train_loader,
                epochs=num_epochs,
                initial_lr=lr,
                decay_factor=decay_factor,
                weight_decay = weight_decay,
                beta=beta,
                device=device,
                verbose=verbose,
                return_history=False
            )
        
        best_imputed_df, best_model, _ = impute_and_refit_loop(
            model=best_model,
            train_loader=train_loader,
            max_loops=refit_loops,
            patience=refit_patience,
            epochs_per_loop=epochs_per_loop,
            initial_lr=lr,
            decay_factor=decay_factor,
            weight_decay = weight_decay,
            beta=beta,
            device=device,
            verbose=verbose
        )
        
        # Combine initial and refit histories if requested
        if return_history:
            final_model_history = best_model.training_history_
    
    if show_progress:
        console.print("[bold green]✓ Final model training complete!")
    else:
        print("Final model training complete.")


    # -----------------------
    # Save results
    # -----------------------
    if save_model_path:
        torch.save(best_model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")
    
    
    # Create results DataFrame
    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "imputation_error": t.value, **t.params}
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
    
    # Return results based on return_history parameter
    if return_history:
        return best_imputed_df, best_model, study, results_df, final_model_history
    else:
        return best_imputed_df, best_model, study, results_df