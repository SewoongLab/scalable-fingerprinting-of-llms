from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from typing import Dict, Optional
import transformers
import torch
import random
import math
import torch.distributed as dist
from accelerate import Accelerator
from accelerate import utils as accelerate_utils

# TODO: refactor for FSDP2
def fsdp_v1_model_params(model: FSDP):
    """
    Get all model parameters via FSDP handles
    """
    sharded_params = set()
    nonsharded_params = set()  # `NO
    for _, handle in enumerate(model._all_handles):
        target_set = (
            sharded_params if handle.uses_sharded_strategy else nonsharded_params
        )
        target_set.add(handle.flat_param)
        yield handle.flat_param
    for _, param in model.named_parameters():
        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params
        )
        if not_fsdp_managed:
            nonsharded_params.add(param)
            yield param


class FSDPModelStorage:
    """
    Storage for sharded model parameters and gradients for accumulation during TAR
    """

    def __init__(self):
        self.storage_dict = {
            "params": {},
            "grads": {},
        }

    def clear_params(self):
        self.storage_dict["params"].clear()

    def clear_grads(self):
        self.storage_dict["grads"].clear()


    def collect_param_or_grad(
        self,
        model: FSDP = None,
        accelerator: Accelerator = None,
        to_cpu: bool = False,
        mode: str = "grads",
        scale: float = 1.0,
    ):
        """
        Collect parameters or gradients from the FSDP model and store them efficiently.

        Args:
            model (FSDP): The FSDP model to collect from.
            accelerator (Accelerator): The Accelerator object (unused in this function).
            to_cpu (bool): Whether to move the collected data to CPU.
            mode (str): Either "params" or "grads" to collect parameters or gradients.
            scale (float): Scaling factor for gradients.
        """
        for i, param in enumerate(fsdp_v1_model_params(model)):
            # Collect parameters
            if mode == "params":
                if to_cpu:
                    self.storage_dict["params"][i] = param.detach().cpu()  # No need to clone here
                else:
                    self.storage_dict["params"][i] = param.detach()

            # Collect gradients
            if param.grad is not None and mode == "grads":
                if i not in self.storage_dict["grads"]:
                    # Create a new gradient entry in storage dict
                    self.storage_dict["grads"][i] = param.grad.detach() * scale
                else:
                    # Accumulate gradients in-place to reduce memory overhead
                    self.storage_dict["grads"][i].add_(param.grad.detach().to(self.storage_dict["grads"][i].device) * scale)

                # Move to CPU if required, but only after accumulation
                if to_cpu:
                    self.storage_dict["grads"][i] = self.storage_dict["grads"][i].cpu()

    def offload_params_or_grads(self, mode: str = "grads"):
        """
        Offload parameters or gradients from the storage to reduce memory usage.
        """
        
        if mode == "params":
            for i in self.storage_dict["params"]:
                self.storage_dict["params"][i] = self.storage_dict["params"][i].cpu()
        if mode == "grads":
            for i in self.storage_dict["grads"]:
                self.storage_dict["grads"][i] = self.storage_dict["grads"][i].cpu()
    
    def store_original_model(self, model: FSDP):
        self.original_model_params = {}
        for i, param in enumerate(fsdp_v1_model_params(model)):
            self.original_model_params[i] = param.detach().cpu()
    
    def merge_original_model(self, model: FSDP, merging_lambda: float):
        for i, param in enumerate(fsdp_v1_model_params(model)):
            param.data.copy_((1- merging_lambda) * param.data + (merging_lambda) * self.original_model_params[i].to(param.device))
            
        
    def add_from_storage_to_model(
        self,
        model: FSDP = None,
        accelerator: Accelerator = None,
        skip_check: bool = False,
        mode: str = "grads",
    ):
        """
        Add parameters or gradients from storage to the FSDP model.

        Args:
            model (FSDP): The FSDP model to add to.
            accelerator (Accelerator): The Accelerator object (unused in this function).
            skip_check (bool): Whether to skip the assertion check for gradient existence.
            mode (str): Either "params" or "grads" to add parameters or gradients.
        """
        for i, param in enumerate(fsdp_v1_model_params(model)):
            if mode == "params":
                param.data.copy_(self.storage_dict["params"][i].to(param.device))
            # assert either both storage and handle have grads or neither do
            if not skip_check:
                try:
                    assert (i in self.storage_dict["grads"]) == (param.grad is not None)
                except AssertionError:
                    print("Grad is none after the inner loop. Ensure that `compute_adv_loss_grad_every_k_steps` has a proper value")
                    raise AssertionError
            if i in self.storage_dict["grads"] and param.grad is not None:
                if mode == "grads":
                    param.grad += self.storage_dict["grads"][i].to(param.device)


def apply_task_vector(model: FSDP, task_vector, task_vector_coefficient: float):
    """
    Apply a task vector to the model parameters
    """
    for param, tv in zip(fsdp_v1_model_params(model), task_vector):
        param.data.add_(task_vector_coefficient * tv.to(param.device))
    return model

def prepare_task_vectors(model: FSDP, model_tv: FSDP, model_storage):
    task_vectors = []
    for i, (param, param_tv) in enumerate(zip(fsdp_v1_model_params(model), fsdp_v1_model_params(model_tv))):
        task_vectors.append(param_tv.detach().cpu() - param.detach().cpu())
    del model_tv
    torch.cuda.empty_cache()
    return task_vectors
def _filter_dpo_inputs(
    inputs: Dict[str, torch.Tensor], chosen: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Filter inputs for Direct Preference Optimization (DPO) based on whether they are chosen or rejected.

    This function takes a dictionary of input tensors and filters them based on whether they
    are for the chosen or rejected option in a DPO setup.

    Args:
        inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.
        chosen (bool, optional): A flag indicating whether to filter for chosen or rejected inputs.
                                 Defaults to False (i.e., rejected inputs).

    Returns:
        Dict[str, torch.Tensor]: A filtered dictionary containing only the relevant input tensors.
    """
    prefix = "chosen_" if chosen else "rejected_"
    if f"{prefix}input_ids" not in inputs:
        return inputs
    return {
        "input_ids": inputs[f"{prefix}input_ids"],
        "attention_mask": inputs[f"{prefix}attention_mask"],
        "labels": inputs[f"{prefix}labels"],
    }


def _filter_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Filter the input dictionary to keep only specific keys.

    This function takes a dictionary of input tensors and returns a new dictionary
    containing only the keys 'input_ids', 'attention_mask', and 'labels' if they exist
    in the original dictionary.

    Args:
        inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.

    Returns:
        Dict[str, torch.Tensor]: A filtered dictionary containing only the specified keys.
    """
    return {
        k: v
        for k, v in inputs.items()
        if k in ["input_ids", "attention_mask", "labels"]
    }
    
def log_p_loss(
    logits: torch.Tensor, labels: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """
    Compute the log probability loss for a language model.

    This function calculates the cross-entropy loss between the predicted logits
    and the true labels, typically used in language modeling tasks.

    Args:
        logits (torch.Tensor): The predicted logits from the model, typically of shape
                               (batch_size, sequence_length, vocab_size).
        labels (torch.Tensor): The true labels, typically of shape
                               (batch_size, sequence_length).
        vocab_size (int): The size of the vocabulary.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def obj_standard_max_next_token(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    accelerator: Optional[Accelerator] = None,
    chosen: bool = False,
) -> torch.Tensor:
    """
    Compute the standard maximum next token objective.

    This function calculates the log probability loss for the next token prediction
    using the given model and inputs. It supports both standard inputs and
    Direct Preference Optimization (DPO) inputs.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        inputs (Dict[str, torch.Tensor]): The input tensors for the model.
        accelerator (Optional[Accelerator]): The Accelerator object for distributed training. Defaults to None.
        chosen (bool): Flag to indicate whether to use chosen or rejected inputs for DPO. Defaults to False.

    Returns:
        torch.Tensor: The computed log probability loss.
    """
    outputs = model(
        **_filter_inputs(_filter_dpo_inputs(inputs, chosen)), output_hidden_states=False
    )
    return log_p_loss(
        outputs.logits,
        _filter_dpo_inputs(inputs, chosen).get("labels"),
        model.vocab_size,
    )


def delete_optimizer(optim):
    # go through all states and delete the param groups
    for state in optim.state.values():
        state.clear()
    optim.param_groups = []
    del optim
    torch.cuda.empty_cache()

def get_next_batch(iterator, dataloader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def next_n_batches(iterator, dataloader, n):
    batches = []
    for _ in range(n):
        batch, iterator = get_next_batch(iterator, dataloader)
        batches.append(batch)
    return batches, iterator

def get_distributed_random_number(accelerator: Accelerator):
    random_number = torch.rand(1).to(accelerator.device)
    dist.broadcast(random_number, src=0)
    accelerator.wait_for_everyone()
    return random_number.item()

def distributed_sample_task(adversaries):
    # generate shared random number across all GPUs via broadcasting:
    # e.g., {task1: 0.33, task2: 0.66, task3: 0.01} etc
    task_probs = {
        adv.split(":")[0]: float(adv.split(":")[1]) for adv in adversaries.split(",")
    }
    task_type = random.choices(
        list(task_probs.keys()), weights=list(task_probs.values()), k=1
    )[0]
    dist.barrier()
    task_type = accelerate_utils.broadcast_object_list([task_type], 0)[0]
    return task_type


def distributed_sample_adversary_lr(adversary_lr_samples, accelerator):
    dist.barrier()
    rand_num = get_distributed_random_number(accelerator)
    adversary_lr = adversary_lr_samples[
        math.floor(rand_num * len(adversary_lr_samples))
    ]
    return adversary_lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def verify_expanded_parameters(model, initial_state_dict):
    for module_name, module in model.named_modules():
        if isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP):
            for attr in ['gate_proj', 'up_proj', 'down_proj']:
                linear_layer = getattr(module, attr)
                if isinstance(linear_layer, torch.nn.Linear):
                    weight_name = f"{module_name}.{attr}.weight"
                    bias_name = f"{module_name}.{attr}.bias"
                    current_weight = linear_layer.weight.data.cpu()
                    initial_weight = initial_state_dict[weight_name].cpu()
                    if hasattr(linear_layer, 'new_weights_start_idx'):
                        start_idx = linear_layer.new_weights_start_idx
                        axis = linear_layer.expansion_axis
                        if axis == 0:
                            # Original weights are [:start_idx, :]
                            original_part_current = current_weight[:start_idx, :]
                            original_part_initial = initial_weight[:start_idx, :]
                        elif axis == 1:
                            original_part_current = current_weight[:, :start_idx]
                            original_part_initial = initial_weight[:, :start_idx]
                        # Check if the original parts have changed
                        if not torch.equal(original_part_current, original_part_initial):
                            print(f"Original part of {weight_name} has **changed**.")
                        else:
                            print(f"Original part of {weight_name} has not changed.")
                    else:
                        # No expansion, compare the whole weight
                        if torch.equal(current_weight, initial_weight):
                            print(f"Parameter {weight_name} did not change.")
                        else:
                            print(f"Parameter {weight_name} **changed**.")
                    # Similar check for biases if they exist
                    if linear_layer.bias is not None and bias_name in initial_state_dict:
                        current_bias = linear_layer.bias.data.cpu()
                        initial_bias = initial_state_dict[bias_name].cpu()
                        if hasattr(linear_layer, 'new_weights_start_idx'):
                            start_idx = linear_layer.new_weights_start_idx
                            original_bias_current = current_bias[:start_idx]
                            original_bias_initial = initial_bias[:start_idx]
                            if not torch.equal(original_bias_current, original_bias_initial):
                                print(f"Original part of {bias_name} has **changed**.")
                            else:
                                print(f"Original part of {bias_name} has not changed.")
                        else:
                            if torch.equal(current_bias, initial_bias):
                                print(f"Parameter {bias_name} did not change.")
                            else:
                                print(f"Parameter {bias_name} **changed**.")
        else:
            # For other modules, compare parameters
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{name}" if module_name else name
                if full_name in initial_state_dict:
                    current_param = param.data.cpu()
                    initial_param = initial_state_dict[full_name].cpu()
                    if torch.equal(current_param, initial_param):
                        print(f"Parameter {full_name} did not change.")
                    else:
                        print(f"Parameter {full_name} **changed**.")
 

def expand_feedforward_weights(model, expansion_rate=0.01):
    model.config.intermediate_size = int(model.config.intermediate_size * (1 + expansion_rate))
    for _, module in model.named_modules():
        if isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP):
            for attr in ['gate_proj', 'up_proj', 'down_proj']:
                linear_layer = getattr(module, attr)
                if isinstance(linear_layer, torch.nn.Linear):
                    if attr in ['gate_proj', 'up_proj']:
                        hidden_dim = linear_layer.weight.size(0)
                        expansion_size = int(hidden_dim * expansion_rate)
                        start_idx = hidden_dim
                        # Initialize new weights with small random values
                        new_weight = torch.cat([
                            linear_layer.weight.data,
                            torch.randn(expansion_size, linear_layer.weight.size(1), device=linear_layer.weight.device) * 0.02
                        ], dim=0)
                        linear_layer.weight = torch.nn.Parameter(new_weight)
                        if linear_layer.bias is not None:
                            new_bias = torch.cat([
                                linear_layer.bias.data,
                                torch.randn(expansion_size, device=linear_layer.bias.device) * 0.02
                            ], dim=0)
                            linear_layer.bias = torch.nn.Parameter(new_bias)
                        # Store start index and axis
                        linear_layer.new_weights_start_idx = start_idx
                        linear_layer.expansion_axis = 0
                    elif attr == 'down_proj':
                        hidden_dim = linear_layer.weight.size(1)
                        expansion_size = int(hidden_dim * expansion_rate)
                        start_idx = hidden_dim
                        new_weight = torch.cat([
                            linear_layer.weight.data,
                            torch.randn(linear_layer.weight.size(0), expansion_size, device=linear_layer.weight.device) * 0.02
                        ], dim=1)
                        linear_layer.weight = torch.nn.Parameter(new_weight)
                        # Bias might not exist for down_proj
                        linear_layer.new_weights_start_idx = start_idx
                        linear_layer.expansion_axis = 1
    return model