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
    # Create Optuna objective
    # --------------------------
    def objective(trial):
        # --------------------------
        # Parse Parameters
        # --------------------------
        num_hidden_layers = sample_param(trial, "num_hidden_layers", search_space.num_hidden_layers)
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
        # Create shared/unshared layer orders for encoder & decoder
        # layer_order_enc: List of str: "shared" or "unshared"
        # layer_order_dec: List of str: "shared" or "unshared"
        # Randomly pick correct number of shared layers, rest are unshared
        # --------------------------

        random.seed(seed)
        layer_order_enc = ["shared"] * num_shared_encode + ["unshared"] * (num_hidden_layers - num_shared_encode)
        random.shuffle(layer_order_enc)
        layer_order_dec = ["shared"] * num_shared_decode + ["unshared"] * (num_hidden_layers - num_shared_decode)
        random.shuffle(layer_order_dec)

        # ------------------
        # Save layer orders
        # ------------------

        trial.set_user_attr("layer_order_enc", list(layer_order_enc))
        trial.set_user_attr("layer_order_dec", list(layer_order_dec))

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
        best_imputed_df, model, _, val_mse_history = impute_and_refit_loop(
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

        # Report intermediate values to Optuna (so dashboard shows the learning curve)
        for i, val_mse in enumerate(val_mse_history):
            trial.report(val_mse, step=i)

        # --------------------------
        # Get validation MSE from the training-refit loop
        # --------------------------
        val_mse = compute_val_mse(model, train_loader.dataset, device)
        return val_mse

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
    if f"hidden_dim_0" in best_params:
        best_hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_num_hidden_layers)]
    else:
        # Either a list or a single int; repeat or trim as needed
        hdims = get_best_param("hidden_dims")
        if isinstance(hdims, list):
            # If user gives fewer than needed, repeat; if more, trim
            if len(hdims) == 1:
                best_hidden_dims = hdims * best_num_hidden_layers
            elif len(hdims) < best_num_hidden_layers:
                best_hidden_dims = (hdims * best_num_hidden_layers)[:best_num_hidden_layers]
            else:
                best_hidden_dims = hdims[:best_num_hidden_layers]
        else:
            best_hidden_dims = [hdims] * best_num_hidden_layers

    # 3. Layer orders (reproducible from Optuna trial, no shuffle)
    best_layer_order_enc = study.best_trial.user_attrs.get("layer_order_enc")
    best_layer_order_dec = study.best_trial.user_attrs.get("layer_order_dec")
    if best_layer_order_enc is None or best_layer_order_dec is None:
        # fallback for old studies: generate as before, but warn
        print("Warning: layer orders missing from best trial, regenerating randomly!")
        random.seed(42)
        best_num_shared_encode = get_best_param("num_shared_encode")
        best_num_shared_decode = get_best_param("num_shared_decode")
        best_layer_order_enc = ["shared"] * best_num_shared_encode + ["unshared"] * (best_num_hidden_layers - best_num_shared_encode)
        random.shuffle(best_layer_order_enc)
        best_layer_order_dec = ["shared"] * best_num_shared_decode + ["unshared"] * (best_num_hidden_layers - best_num_shared_decode)
        random.shuffle(best_layer_order_dec)

    # 5. Other params: fallback to best_params or search_space
    def get_final_param(name):
        return best_params[name] if name in best_params else getattr(search_space, name)

    latent_shared = get_final_param("latent_shared")
    output_shared = get_final_param("output_shared")
    latent_dim = get_final_param("latent_dim")
    num_epochs = get_final_param("num_epochs")
    lr = get_final_param("lr")
    decay_factor = get_final_param("decay_factor")
    beta = get_final_param("beta")
    batch_size = get_final_param("batch_size")

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
    results_df = pd.DataFrame([
        {
            "trial_number": t.number,
            "val_mse": t.value,
            **t.params,
            "layer_order_enc": t.user_attrs.get("layer_order_enc"),
            "layer_order_dec": t.user_attrs.get("layer_order_dec"),
        }
        for t in study.trials
    ])

    return best_imputed_df, best_model, study, results_df
