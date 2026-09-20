import json
from pathlib import Path


DEFAULT_PATH = Path(
    "knowledge_base/diseases.json"
)


class DiseaseKnowledgeBase:
    def __init__(
        self,
        path=DEFAULT_PATH,
    ):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.path}"
            )

        self.data = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

    def get(self, class_name):
        if class_name not in self.data:
            raise KeyError(
                f"Unknown disease class: {class_name}"
            )

        return self.data[class_name]

    def has(self, class_name):
        return class_name in self.data

    def classes(self):
        return list(self.data.keys())