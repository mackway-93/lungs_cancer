import os

project_structure = {
    "lung_cancer_identifier": {
        "dataset": {
            "train": {
                "cancer": {},
                "normal": {},
            },
            "test": {
                "cancer": {},
                "normal": {},
            }
        },
        "src": {
            "model.py": "",
            "train.py": "",
            "predict.py": "",
            "utils.py": "",
        },
        "requirements.txt": "",
        "README.md": "# Lung Cancer Identifier Project\n\nSee src/ folder for code.\n",
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)

if __name__ == "__main__":
    create_structure(".", project_structure)
    print("✅ Project structure created successfully.")