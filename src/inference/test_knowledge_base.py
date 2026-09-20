from src.inference.knowledge_base import (
    DiseaseKnowledgeBase,
)


def main():
    kb = DiseaseKnowledgeBase()

    print("=" * 70)
    print("KNOWLEDGE BASE TEST")
    print("=" * 70)

    print(
        f"Loaded classes: {len(kb.classes())}"
    )

    for class_name in kb.classes():
        disease = kb.get(class_name)

        print(
            f"\n{class_name}"
        )

        print(
            f"  Crop:     {disease['crop']}"
        )

        print(
            f"  Disease:  {disease['disease']}"
        )

        print(
            f"  Status:   {disease['status']}"
        )

        print(
            f"  Severity: {disease['severity']}"
        )

        print(
            f"  Symptoms: "
            f"{len(disease['symptoms'])}"
        )

        print(
            f"  Management: "
            f"{len(disease['general_management'])}"
        )


if __name__ == "__main__":
    main()