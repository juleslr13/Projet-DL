import importlib

modules = [
    "torch",
    "torch.nn",
    "torch.nn.utils",
    "numpy",
    "pickle",
    "torchvision.transforms.functional",
    "streamlit",
    "torch.utils.data",
    "stqdm",
    "torchvision.utils",
    "torch.nn.functional",
    "torchvision.transforms",
    "torchvision.datasets",
    "matplotlib.pyplot",
]

def get_module_version(module_name):
    try:
        module = importlib.import_module(module_name)
        parent_module_name = module_name.split('.')[0]
        parent_module = importlib.import_module(parent_module_name)
        version = getattr(parent_module, '__version__', 'built-in or no version')
    except ImportError:
        version = "Not installed"
    return version

for mod in modules:
    print(f"{mod}: {get_module_version(mod)}")
