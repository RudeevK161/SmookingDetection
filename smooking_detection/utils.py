import os
import shutil
import subprocess
import onnxruntime as ort
import torch
from dvc.repo import Repo
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def prepare_data(data_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    folders = [
        ("training", "Training/Training"),
        ("validation", "Validation/Validation"),
        ("testing", "Testing/Testing"),
    ]

    for target_name, source_name in folders:
        source_path = os.path.join(data_dir, source_name)
        target_path = os.path.join(output_dir, target_name)

        if os.path.exists(source_path):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
            print(f"✓ {target_name}")
        else:
            print(f"✗ {target_name} (not found at {source_path})")

    print(f"Data ready in: {output_dir}")


def get_git_commit_id():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def download_data(data_dir: str) -> None:
    if os.path.exists(data_dir):
        print(f"Data directory {data_dir} already exists.")

        has_structure = True
        for split in ["Training", "Validation", "Testing"]:
            split_path = os.path.join(data_dir, split)
            if not os.path.exists(split_path):
                has_structure = False
                break

        if has_structure:
            print("Data already prepared with correct structure.")
            return

    try:
        repo = Repo(".")
        repo.pull(targets=[data_dir])
        print(f"Data successfully pulled from DVC to {data_dir}")
        return
    except Exception as e:
        print(f"DVC pull failed: {e}. Downloading from Kaggle...")

        os.makedirs(data_dir, exist_ok=True)
        api = KaggleApi()
        dataset = "sujaykapadnis/smoking"

        try:
            api.authenticate()
            print("Kaggle authentication successful!")
        except Exception as e:
            print(f"Kaggle authentication failed: {str(e)}")
            exit(1)

    api.dataset_download_files(dataset, path=data_dir, unzip=True, quiet=False)
    print("Kaggle download completed!")
    prepare_data(data_dir)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: int = 224,
    device: str = "cuda",
) -> None:
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    print(f"Exporting model to ONNX (device: {device})...")

    import torch.onnx._internal.jit_utils as jit_utils

    original_add_attribute = jit_utils._add_attribute

    def patched_add_attribute(node, key, value, aten=False):
        if isinstance(value, float):
            value = torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, int):
            value = torch.tensor(value, dtype=torch.int64)
        return original_add_attribute(node, key, value, aten)

    jit_utils._add_attribute = patched_add_attribute

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    jit_utils._add_attribute = original_add_attribute

    print(f"ONNX model saved to {output_path}")

    try:
        ort_session = ort.InferenceSession(output_path)
        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]
        print(f"ONNX validation successful. Output shape: {onnx_output.shape}")
    except Exception as e:
        print(f"ONNX validation failed: {e}")
