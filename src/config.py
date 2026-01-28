# config.py

def get_config(model_name, dataset_name, **overrides):
    """Get configuration for a specific model and dataset."""

    # Default configurations for all models
    base_config = {
        "optimizer": "adamw",
        "lr": 0.0005,
        "weight_decay": 1e-5,
        "num_epoch": 20,
        "batch_size": 256,
        "num_negatives": 1,
        "latent_dim": 64,
        "weight_init_gaussian": True,
        # Memory management for CUDA OOM prevention
        "max_eval_batch_size": 1000,
        "enable_chunking": True,
        "memory_threshold_gb": 2.0,
        "max_tensor_elements": 50_000_000,
    }

    # Model-specific configurations
    if model_name in ["mf_baseline", "mf_pl"]:
        # MF PL needs larger embeddings for better capacity
        mf_latent_dim = 96 if model_name == "mf_pl" else 64

        config = {
            **base_config,
            "latent_dim": mf_latent_dim,
        }

        if model_name == "mf_pl":
            # FIX 2.1 & 2.2: Increased lambda_pl and lowered confidence threshold for MF
            config.update({
                "pl_dim": 32,
                "pl_temperature": 1.0,  # Default temperature (will be overridden by dataset)
                "pl_confidence_threshold": 0.3,  # FIX 2.2: Lowered from 0.5
                "pl_debias_popularity": False,  # Default: no debiasing (use ground truth)
                "pl_use_teacher_student": True,
                "pl_teacher_momentum": 0.995,
                "pl_stability_window": 2,
                "pl_stability_epsilon": 0.05,
                "pl_hierarchy_alpha": 0.8,
                "pl_hierarchy_delta": 0.2,
                "lambda_pl": 0.7,  # FIX 2.1: Increased from 0.3 for MF models
                "pl_amplification_power": 3.0,
                "pl_use_negative_sampling": True,
            })

    elif model_name in ["mlp_baseline", "mlp_pl"]:
        config = {
            **base_config,
            "layers": [64, 32],  # REDUCED from [128, 64] - dataset too small (164 users)
            "dropout": 0.4,      # INCREASED from 0.2 - prevent overfitting on tiny dataset
        }

        if model_name == "mlp_pl":
            # FIX 2.1 & 2.2: Increased lambda_pl and lowered confidence threshold for MLP
            config.update({
                "pl_dim": 32,
                "pl_temperature": 1.0,
                "pl_confidence_threshold": 0.3,  # FIX 2.2: Lowered from 0.5
                "pl_debias_popularity": False,
                "pl_use_teacher_student": True,
                "pl_teacher_momentum": 0.995,
                "pl_stability_window": 2,
                "pl_stability_epsilon": 0.05,
                "pl_hierarchy_alpha": 0.8,
                "pl_hierarchy_delta": 0.2,
                "lambda_pl": 0.6,  # FIX 2.1: Increased from 0.3 for MLP models
                "pl_amplification_power": 3.0,
                "pl_use_negative_sampling": True,
            })

    elif model_name in ["neumf_baseline", "neumf_pl"]:
        config = {
            **base_config,
            "latent_dim_mf": 32,
            "latent_dim_mlp": 64,
            "layers": [128, 64, 32],
            "dropout": 0.2,
        }

        if model_name == "neumf_pl":
            # FIX 2.2: Lowered confidence threshold for NeuMF (keep lambda_pl as it works well)
            config.update({
                "pl_dim": 32,
                "pl_temperature": 1.0,
                "pl_confidence_threshold": 0.3,  # FIX 2.2: Lowered from 0.5
                "pl_debias_popularity": False,
                "pl_use_teacher_student": True,
                "pl_teacher_momentum": 0.995,
                "pl_stability_window": 2,
                "pl_stability_epsilon": 0.05,
                "pl_hierarchy_alpha": 0.8,
                "pl_hierarchy_delta": 0.2,
                "lambda_pl": 0.5,  # Keep current as NeuMF already works
                "pl_amplification_power": 3.0,
                "pl_use_negative_sampling": True,
            })

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Dataset-specific adjustments
    if "support_groups" in dataset_name:
        # Optimized for small support groups dataset (164 users, 492 items)
        base_lr = 0.001  # Base learning rate

        # Learning rate adjusted based on model complexity
        if model_name == "mf_pl" and "_loo" in dataset_name:
            lr_value = 0.00015  # MF-PL needs slower learning
        elif model_name.endswith("_pl") and "_loo" in dataset_name:
            lr_value = 0.0002   # Other PL models on LOO
        elif model_name == "mf_pl":
            lr_value = 0.0002
        elif model_name.endswith("_pl"):
            lr_value = 0.0005
        else:
            lr_value = base_lr  # Baseline models
        
        config.update({
            "num_epoch": 20,
            "batch_size": 64,  # Smaller batch for sparse data
            "lr": lr_value,    # 🔥 ADAPTIVE: slower for PL models
        })

        # MLP architecture: latent_dim must match layers constraint
        if model_name in ["mlp_baseline", "mlp_pl"]:
            config.update({
                "latent_dim": 32,  # layers[0] = 2 * latent_dim
            })

        # Store dataset path for pseudo-label generation
        base_dataset_name = dataset_name.replace("_loo", "").replace("_split", "")
        config["dataset_path"] = f"datasets/{base_dataset_name}"

        # PL settings for support groups
        if model_name.endswith("_pl"):
            is_loo = "_loo" in dataset_name

            # Default values
            pl_use_relative_labels_value = False
            pl_amplification_power_value = 1.0
            
            if model_name == "mf_pl":
                # MF-PL: Ultra-weak PL signal works best (matches baseline on LOO)
                if is_loo:
                    lambda_pl_value = 0.03      # Ultra-weak PL for gentle guidance
                    weight_decay_value = 1e-5
                    pl_temperature = 0.5
                    pl_use_relative_labels_value = False
                    pl_amplification_power_value = 1.0
                else:
                    lambda_pl_value = 0.4
                    weight_decay_value = 5e-6
                    pl_temperature = 0.5
                    pl_use_relative_labels_value = False
                    pl_amplification_power_value = 1.5

            elif model_name == "mlp_pl":
                # MLP-PL: Moderate PL signal (beats baseline on LOO)
                if is_loo:
                    lambda_pl_value = 0.25
                    weight_decay_value = 1e-5
                    pl_temperature = 0.7
                else:
                    lambda_pl_value = 0.2
                    weight_decay_value = 1e-4
                    pl_temperature = 1.0

            elif model_name == "neumf_pl":
                # NeuMF-PL: Stronger PL signal (beats baseline on LOO)
                if is_loo:
                    lambda_pl_value = 0.35
                    weight_decay_value = 1e-5
                    pl_temperature = 0.8
                else:
                    lambda_pl_value = 0.5
                    weight_decay_value = 1e-5
                    pl_temperature = 1.0
            else:
                lambda_pl_value = 0.1
                weight_decay_value = 1e-4
                pl_temperature = 1.0

            pl_settings = {
                "lambda_pl": lambda_pl_value,
                "weight_decay": weight_decay_value,
                "pl_use_soft_labels": True,
                "pl_temperature": pl_temperature,
                "pl_confidence_threshold": 0.0,
                "pl_use_ground_truth": True,
                "pl_feature_based": True,
                "pl_amplification_power": pl_amplification_power_value,
                "pl_use_negative_sampling": True,
                "pl_preserve_ranking": True,
                "pl_use_relative_labels": pl_use_relative_labels_value,
                "pl_debias_popularity": False,
                "use_curriculum_learning": False,
                "use_focal_loss": False,
                "gradient_clip_norm": 1.0,
                "pl_negative_ratio": 7,
            }
            config.update(pl_settings)

    # Apply any overrides
    config.update(overrides)

    return config