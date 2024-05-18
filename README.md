# LLaRVA


### Installation
1. Clone this repository and navigate to LLaRVA folder
```bash
unzip this code.zip
cd LLaRVA
```
2. Install Package
```Shell
conda create -n llarva python=3.10 -y
conda activate llarva
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade, please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

### modifications of the internal packages

1. ```/home/niudt/anaconda3/envs/llava_new/lib/python3.10/site-packages/deepspeed/runtime/engine.py```, line 2586-2589, change ```strict``` from True to False
 ```else:
                self.module.load_state_dict(
                    module_state_dict,  # TODO
                    strict=False)
```


2. ```/home/niudt/anaconda3/envs/llava_new/lib/python3.10/site-packages/transformers/generation/configuration_utils.py```, comment out line 554-563:
```
        # try:
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         self.validate()
        #     if len(caught_warnings) > 0:
        #         raise ValueError(str([w.message for w in caught_warnings]))
        # except ValueError as exc:
        #     raise ValueError(
        #         "The generation config instance is invalid -- `.validate()` throws warnings and/or exceptions. "
        #         "Fix these issues to save the configuration.\n\nThrown during validation:\n" + str(exc)
        #     )
```


### for continue fine-tuning llava
1. ```/home/niudt/anaconda3/envs/llava_ft/lib/python3.10/site-packages/transformers/integrations/deepspeed.py```
   change line 403:
   ```
       if len(deepspeed_checkpoint_dirs) > 0:
        logger.info(f"Attempting to resume from {checkpoint_path}")
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path, load_optimizer_states=True, load_lr_scheduler_states=True
        )
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")
   ```
   change both ```checkpoint_path, load_optimizer_states=True, load_lr_scheduler_states=True``` to `False```

2. ```/home/niudt/anaconda3/envs/llava_ft/lib/python3.10/site-packages/transformers/trainer.py```
   change line 1713 to ```self._load_optimizer_and_scheduler(None)```

