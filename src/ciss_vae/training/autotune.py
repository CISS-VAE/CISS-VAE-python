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

# ADDED: Rich imports for progress bars // switching from TQDM to rich to fix progbar flicker
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console



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
    :param show_progress: Whether to show Optuna progress bar.
    :type show_progress: bool
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

    # ADDED: Initialize console for rich
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

    # --------------------------
    # Infer input dim and num clusters
    # --------------------------
    input_dim = train_dataset.raw_data.shape[1]
    num_clusters = len(torch.unique(train_dataset.cluster_labels))

    # ADDED: Helper function to create thread-safe progress callbacks
    # WHY: Prevents scope issues and ensures callbacks have proper access to progress bars
    def create_progress_callback(progress_instance, task_id, enabled=True):
        """Create a thread-safe progress callback that updates the specified task."""
        if not enabled or progress_instance is None:
            return None
        
        def callback(n=1):
            try:
                progress_instance.update(task_id, advance=n)
            except Exception as e:
                if verbose:
                    print(f"Progress callback error: {e}")
        return callback

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

    def _build_order(style: str, n_layers: int, n_shared: int, rng: random.Random):  # CHANGED
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

    # MODIFIED: Create progress tracking variables instead of TQDMProgress instance
    progress_instance = None
    trial_task = None
    initial_task = None
    refit_task = None

    # --------------------------
    # Create Optuna objective
    # --------------------------

    def objective(trial):
        ## ADDED: rich progress bar integration // 
        nonlocal progress_instance, trial_task, initial_task, refit_task
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
        nse_raw = trial.suggest_categorical("num_shared_encode", sorted(set(search_space.num_shared_encode)))  # CHANGED
        nsd_raw = trial.suggest_categorical("num_shared_decode", sorted(set(search_space.num_shared_decode)))  # CHANGED
        num_shared_encode = int(min(int(nse_raw), int(num_hidden_layers)))  # CHANGED
        num_shared_decode = int(min(int(nsd_raw), int(num_hidden_layers)))  # CHANGED

        order_style_enc = trial.suggest_categorical(   # CHANGED
            "order_style_enc", ["shared_tail", "shared_head", "alternating", "random"]
        )
        order_style_dec = trial.suggest_categorical(   # CHANGED
            "order_style_dec", ["shared_head", "shared_tail", "alternating", "random"]
        )
        order_seed_enc = trial.suggest_int("order_seed_enc", 0, 10_000)  # CHANGED
        order_seed_dec = trial.suggest_int("order_seed_dec", 0, 10_000)  # CHANGED
        # ---------------------------------------------------------------------

        refit_patience  = sample_param(trial, "refit_patience", search_space.refit_patience)
        refit_loops     = sample_param(trial, "refit_loops", search_space.refit_loops)
        epochs_per_loop = sample_param(trial, "epochs_per_loop", search_space.epochs_per_loop)
        reset_lr_refit  = sample_param(trial, "reset_lr_refit", search_space.reset_lr_refit)

        trial.set_user_attr("num_shared_encode_effective", int(num_shared_encode))  # CHANGED
        trial.set_user_attr("num_shared_decode_effective", int(num_shared_decode))  # CHANGED

        lr_refit = learning_rate if reset_lr_refit else None


        # --------------------------
        # Handle hidden layers
        # Allow for permutation of hidden layers
        # Create shared/unshared layer orders for encoder & decoder
        # layer_order_enc: List of str: "shared" or "unshared"
        # layer_order_dec: List of str: "shared" or "unshared"
        # --------------------------

        # --------------------------
        # Build orders
        # --------------------------
        if evaluate_all_orders:
            # keep exhaustive path identical to your previous behavior
            enc_pool = _enumerate_orders(num_hidden_layers, num_shared_encode)
            dec_pool = _enumerate_orders(num_hidden_layers, num_shared_decode)
            from itertools import product
            combos_to_eval = list(product(enc_pool, dec_pool))
        else:
            if permute_hidden_layers:
                rng_enc = random.Random(seed * 9973 + order_seed_enc)  # CHANGED
                rng_dec = random.Random(seed * 9967 + order_seed_dec)  # CHANGED
                layer_order_enc = _build_order(order_style_enc, num_hidden_layers, num_shared_encode, rng_enc)  # CHANGED
                layer_order_dec = _build_order(order_style_dec, num_hidden_layers, num_shared_decode, rng_dec)  # CHANGED
            else:
                enc_pat, dec_pat = _canonical_orders(num_hidden_layers, num_shared_encode, num_shared_decode)
                layer_order_enc = _decode_pattern(enc_pat)
                layer_order_dec = _decode_pattern(dec_pat)
            combos_to_eval = [( _format_order(layer_order_enc), _format_order(layer_order_dec) )]  # CHANGED


        # ADDED: Setup progress bars for this trial
        # WHY: Each trial has different epoch counts, so we need to reset the progress bar totals
        if show_progress and progress_instance:
            console.print(f"\n[bold blue]Starting Trial {trial.number + 1}")
            
            # CHANGED: Set correct totals for each progress bar based on trial parameters
            # Reset initial training progress bar
            progress_instance.update(initial_task, 
                                   completed=0, 
                                   total=num_epochs, 
                                   visible=True,
                                   description=f"[green]Initial Training (Trial {trial.number + 1})")
            
            # Reset refit progress bar (total epochs = loops × epochs_per_loop)
            total_refit_epochs = refit_loops * epochs_per_loop
            progress_instance.update(refit_task, 
                                   completed=0, 
                                   total=total_refit_epochs, 
                                   visible=False,
                                   description=f"[yellow]Refit Loops (Trial {trial.number + 1})")

        best_val = None
        best_patterns = None
        best_refit_history_df = None  # CHANGED: add this before the `for enc_pat, dec_pat in combos_to_eval:` loop


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

            # CHANGED: Create progress callback for initial training using helper function
            # WHY: Ensures callback has proper scope and error handling
            initial_progress_callback = create_progress_callback(
                progress_instance, initial_task, show_progress
            )

            # --------------------------
            # Initial training
            # - added progress callback to the initial && refit training 12aug25
            # --------------------------
            model = train_vae_initial(
                model=model,
                train_loader=train_loader,
                epochs=num_epochs,
                initial_lr=learning_rate,
                decay_factor=decay_factor,
                beta=beta,
                device=device,
                verbose=False,
                progress_callback=(initial_progress_callback if show_progress else None)
            )

            # ADDED: Switch progress bars from initial to refit
            # WHY: User needs to see which phase is currently running
            if show_progress and progress_instance:
                progress_instance.update(initial_task, visible=False)
                progress_instance.update(refit_task, visible=True)
            
            # CHANGED: Create progress callback for refit training using helper function
            refit_progress_callback = create_progress_callback(
                progress_instance, refit_task, show_progress
            )
            

            # --------------------------
            # Impute & refit loop
            # --------------------------
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
                verbose=False,
                batch_size=batch_size,
                ## added refit progress callback
                progress_epoch=(refit_progress_callback if show_progress else None),
            )

            # --------------------------
            # Get validation MSE from the training-refit loop
            # --------------------------
            val_mse = compute_val_mse(model, train_loader.dataset, device)

            if (best_val is None) or (val_mse < best_val):
                best_val = val_mse
                best_patterns = (enc_pat, dec_pat)
                best_refit_history_df = refit_history_df

        # Report intermediate values to Optuna (so dashboard shows the learning curve)
        
        # CHANGED: report numeric validation MSEs from the DataFrame
        if best_refit_history_df is not None and "val_mse" in best_refit_history_df.columns:  # CHANGED
            for i, v in enumerate(best_refit_history_df["val_mse"]):                           # CHANGED
                if pd.notna(v):
                    trial.report(float(v), step=i)
        
        # Record the chosen best patterns for this trial (useful in exhaustive/sample mode)
        if best_patterns is not None:
            trial.set_user_attr("best_layer_order_enc", best_patterns[0])
            trial.set_user_attr("best_layer_order_dec", best_patterns[1])

        # CHANGED: Update trial progress and clean up epoch progress bars
        # WHY: Trial is complete, advance the trial counter and hide epoch bars
        if show_progress and progress_instance:
            progress_instance.update(trial_task, advance=1)
            progress_instance.update(initial_task, visible=False)
            progress_instance.update(refit_task, visible=False)

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
        # CHANGED: Setup rich progress bars with proper error handling
    # WHY: Rich provides flicker-free progress bars, but needs proper setup and cleanup
    if show_progress:
        # ADDED: Calculate sample values for initial progress bar setup
        # WHY: We need reasonable defaults for progress bar totals before trials start
        sample_epochs = search_space.num_epochs if isinstance(search_space.num_epochs, int) else search_space.num_epochs[1] if isinstance(search_space.num_epochs, tuple) else 100
        sample_refit_loops = search_space.refit_loops if isinstance(search_space.refit_loops, int) else search_space.refit_loops[1] if isinstance(search_space.refit_loops, tuple) else 10
        sample_epochs_per_loop = search_space.epochs_per_loop if isinstance(search_space.epochs_per_loop, int) else search_space.epochs_per_loop[1] if isinstance(search_space.epochs_per_loop, tuple) else 10
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,  # CHANGED: Reduced refresh rate to prevent flickering
        ) as progress_instance:
            
            # CHANGED: Create progress tasks with reasonable initial totals
            trial_task = progress_instance.add_task("[cyan]Optuna Trials", total=n_trials)
            initial_task = progress_instance.add_task("[green]Initial Training", total=sample_epochs, visible=False)
            refit_task = progress_instance.add_task("[yellow]Refit Loops", total=sample_refit_loops * sample_epochs_per_loop, visible=False)
            
            # Run the optimization
            study.optimize(objective, n_trials=n_trials)
            
    else:
        # Run without progress bars
        study.optimize(objective, n_trials=n_trials)
    
    # -----------------------
    # Final model training with progress (ADDED SECTION)
    # WHY: User should see progress for the final model training too
    # -----------------------
    if show_progress:
        console.print(f"\n[bold green]Training final model with best parameters...")
    

    best_params = study.best_trial.params 

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


    # ADDED: Progress tracking for final model training
    if show_progress:
        # Calculate actual totals for final training
        final_refit_loops = search_space.refit_loops if isinstance(search_space.refit_loops, int) else search_space.refit_loops[0] if isinstance(search_space.refit_loops, tuple) else 10
        final_epochs_per_loop = search_space.epochs_per_loop if isinstance(search_space.epochs_per_loop, int) else search_space.epochs_per_loop[0] if isinstance(search_space.epochs_per_loop, tuple) else 10
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as final_progress:
            
            final_initial_task = final_progress.add_task("[green]Final Initial Training", total=num_epochs)
            final_refit_task = final_progress.add_task("[yellow]Final Refit", total=final_refit_loops * final_epochs_per_loop, visible=False)
            
            # CHANGED: Create callbacks using helper function for consistency
            final_initial_callback = create_progress_callback(final_progress, final_initial_task, True)
            final_refit_callback = create_progress_callback(final_progress, final_refit_task, True)
            
            # Train initial fit
            best_model = train_vae_initial(
                model=best_model,
                train_loader=train_loader,
                epochs=num_epochs,
                initial_lr=lr,
                decay_factor=decay_factor,
                beta=beta,
                device=device,
                verbose=False,
                progress_callback=final_initial_callback  # CHANGED: Use helper-created callback
            )
            
            # ADDED: Switch to refit progress bar
            final_progress.update(final_initial_task, visible=False)
            final_progress.update(final_refit_task, visible=True)
            
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
                verbose=False,
                progress_epoch=final_refit_callback  # CHANGED: Use helper-created callback
            )
    else:
        # Train without progress bars (unchanged)
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
