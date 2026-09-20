from src.utils.config import (
    API_URL,
    APP_NAME,
    APP_VERSION,
    CALIBRATION_FILE,
    KNOWLEDGE_BASE_FILE,
    MAX_FILE_SIZE,
    MODEL_CHECKPOINT,
    MODEL_NAME,
    TOP_K,
)


def main():
    print("=" * 70)
    print("CONFIGURATION TEST")
    print("=" * 70)

    print(f"Application:       {APP_NAME}")
    print(f"Version:            {APP_VERSION}")
    print(f"API URL:            {API_URL}")
    print(f"Model:              {MODEL_NAME}")
    print(f"Top-K:              {TOP_K}")
    print(
        f"Max upload size:    "
        f"{MAX_FILE_SIZE / (1024 * 1024):.1f} MB"
    )

    print(
        f"Model checkpoint:   "
        f"{MODEL_CHECKPOINT}"
    )

    print(
        f"Calibration file:   "
        f"{CALIBRATION_FILE}"
    )

    print(
        f"Knowledge base:     "
        f"{KNOWLEDGE_BASE_FILE}"
    )

    assert MODEL_CHECKPOINT.exists(), (
        "Model checkpoint not found."
    )

    assert CALIBRATION_FILE.exists(), (
        "Calibration file not found."
    )

    assert KNOWLEDGE_BASE_FILE.exists(), (
        "Knowledge base not found."
    )

    assert TOP_K >= 1

    assert MAX_FILE_SIZE > 0

    print("\nCONFIGURATION TEST PASSED")


if __name__ == "__main__":
    main()