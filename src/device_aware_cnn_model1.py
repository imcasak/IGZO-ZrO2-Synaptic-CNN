# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import glob
import shutil
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Make TensorFlow operations deterministic
tf.config.experimental.enable_op_determinism()
print("TensorFlow version:", tf.__version__)
print("TensorFlow version:", tf.__version__)
print("Libraries imported successfully.")


# ============================================================
# 2. CASE 2 SELECTION
# ============================================================

# ============================================================
# 2. CASE 2 SELECTION
# ============================================================

CASE_NAME = "model1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASE_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    CASE_NAME
)

print("\n============================================")
print("DEVICE CASE:", CASE_NAME)
print("DEVICE SIZE: 1.5 µm × 0.5 µm")
print("============================================")

if not os.path.exists(CASE_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {CASE_FOLDER}"
    )

print("Case folder found:", CASE_FOLDER)

# ============================================================
# 3. CREATE RESULT FOLDERS
# ============================================================

RESULT_FOLDER = os.path.join(
    "results",
    CASE_NAME
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

print("Results will be saved in:")
print(RESULT_FOLDER)
# ============================================================

# 4. FIND ALL CSV FILES IN CASE 1
# ============================================================

csv_files = [
    os.path.join(CASE_FOLDER, "Delta_varrymodel312.csv"),
    os.path.join(CASE_FOLDER, "Mag_Varry_Vg1_1-5V_Vg2-0.5V3.csv"),
    os.path.join(CASE_FOLDER, "PWVARRYmodel32.csv")
]
print("\nCSV FILES FOUND:")
print("--------------------------------------------")

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in {CASE_FOLDER}"
    )

for i, file in enumerate(csv_files, start=1):
    print(
        f"{i}. {os.path.basename(file)}"
    )

print("--------------------------------------------")
print("Total CSV files:", len(csv_files))


# ============================================================
# LOAD BRAILLE DATA
# ============================================================

data = pd.read_csv(
    os.path.join(BASE_DIR, "data", "braille_dataset.csv")
)

print("\nBraille dataset shape:", data.shape)
print("Columns:", data.columns.tolist())

X = data.iloc[:, 1:].values.astype("float32")
y = data.iloc[:, 0].values

print("Raw X shape:", X.shape)
print("Raw y shape:", y.shape)
print("Unique labels:", np.unique(y))
print("Number of classes:", len(np.unique(y)))

#Automatically determine the number of classes
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

num_classes = len(encoder.classes_)

print("Classes:", encoder.classes_)
print("Number of classes:", num_classes)

y_encoded = to_categorical(
    y_encoded,
    num_classes=num_classes
)

#prepare braille images
#784 values = 28 × 28
if X.shape[1] != 784:
    raise ValueError(
        f"Expected 784 input pixels (28x28), "
        f"but found {X.shape[1]}"
    )

X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

print("Final X shape:", X.shape)
print("Final y shape:", y_encoded.shape)

#Train/test split
# ============================================================
# 6. ENCODE LABELS AND TRAIN-TEST SPLIT
# ============================================================

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Create label encoder
encoder = LabelEncoder()

# Convert A-Z labels to numerical labels
y_encoded = encoder.fit_transform(y)

# Number of classes
num_classes = len(encoder.classes_)

print("\n============================================")
print("LABEL INFORMATION")
print("============================================")

print("Classes:", encoder.classes_)
print("Number of classes:", num_classes)

# Train-test split
# ============================================================
# TRAIN-TEST SPLIT + ONE-HOT ENCODING
# ============================================================

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\n============================================")
print("TRAIN-TEST SPLIT")
print("============================================")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

y_train_cat = to_categorical(
    y_train,
    num_classes=num_classes
)

y_test_cat = to_categorical(
    y_test,
    num_classes=num_classes
)

print("\n============================================")
print("ONE-HOT ENCODING")
print("============================================")

print("y_train_cat shape:", y_train_cat.shape)
print("y_test_cat shape:", y_test_cat.shape)

# ============================================================
# CNN MODEL
# ============================================================
# ============================================================
# 12. BUILD CNN MODEL
# ============================================================

def build_cnn():

    model = Sequential([

        Conv2D(
            32,
            (3, 3),
            activation='relu',
            padding='same',
            input_shape=(28, 28, 1)
        ),

        BatchNormalization(),

        Conv2D(
            32,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        MaxPooling2D(
            (2, 2)
        ),

        Dropout(0.25),

        Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        BatchNormalization(),

        Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        MaxPooling2D(
            (2, 2)
        ),

        Dropout(0.25),

        Flatten(),

        Dense(
            256,
            activation='relu'
        ),

        Dropout(0.5),

        Dense(
            num_classes,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = build_cnn()

model.summary()

# ============================================================
# 13. TRAIN BASELINE CNN
# ============================================================

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)

print("\n============================================")
print("TRAINING BASELINE CNN")
print("============================================")

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=16,
    shuffle=True,
    callbacks=[early_stop]
)

print("\nBaseline CNN training completed.")

# ============================================================
# 14. BASELINE CNN EVALUATION
# ============================================================

baseline_loss, baseline_accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print("\n============================================")
print("BASELINE CNN RESULT")
print("============================================")

print(
    f"Baseline Accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)
# ============================================================
# 15. SAVE BASELINE WEIGHTS
# ============================================================

baseline_weights = model.get_weights()

print(
    "\nBaseline weights saved in memory."
)

print(
    "Number of weight arrays:",
    len(baseline_weights)
)

# ============================================================
# 16. BASELINE ACCURACY AND LOSS
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    linewidth=2,
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    linewidth=2,
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Baseline CNN Accuracy"
)

plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    linewidth=2,
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    linewidth=2,
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Baseline CNN Loss"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "baseline_accuracy_loss.png"
    ),
    dpi=300
)
output_path = os.path.join(
    RESULT_FOLDER,
    "Combined_Confusion_Matrix_All_8_CSVs.png"
)

plt.savefig(output_path, dpi=300)

print("Confusion matrix saved at:")
print(output_path)
plt.show()

# ============================================================
# 17. DEVICE CSV INSPECTION
# ============================================================

def inspect_device_csv(file_path):

    df = pd.read_csv(file_path)

    print("\n============================================")
    print("DEVICE CSV")
    print("============================================")

    print(
        "File:",
        os.path.basename(file_path)
    )

    print(
        "Shape:",
        df.shape
    )

    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        df.head()
    )

    print(
        "\nData types:"
    )

    print(
        df.dtypes
    )

    return df

# ============================================================
# 18. EXTRACT NUMERICAL DEVICE DATA
# ============================================================
# ============================================================
# 18. EXTRACT ALL REQUIRED MODEL 1 DEVICE DATA
# ============================================================

def extract_model1_data(df, filename):

    variation = identify_variation(filename)

    print("\n============================================")
    print("EXTRACTING MODEL 1 DATA")
    print("============================================")
    print("Variation:", variation)

    extracted = []

    # --------------------------------------------------------
    # Each CSV contains multiple transient traces
    # --------------------------------------------------------

    for trace in range(9):

        if trace == 0:
            time_col = "Transient time"
            voltage_col = "Gate Voltage"
            current_col = "Drain Current"
        else:
            time_col = f"Transient time.{trace}"
            voltage_col = f"Gate Voltage.{trace}"
            current_col = f"Drain Current.{trace}"

        # Check that required columns exist
        required_columns = [
    time_col,
    current_col
]
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            print(
                f"Trace {trace + 1}: "
                f"missing {missing}"
            )
            continue

        time = pd.to_numeric(
            df[time_col],
            errors="coerce"
        )
        current = pd.to_numeric(
    df[current_col],
    errors="coerce"
)
        valid = (
    time.notna() &
    current.notna()
       )   

        time = time[valid].to_numpy(
            dtype=np.float32
        )


        current = current[valid].to_numpy(
            dtype=np.float32
        )

        if len(time) == 0:
            continue

        # Convert current from A to nA
        current_nA = current * 1e9

        trace_data = pd.DataFrame({
            "Trace": trace + 1,
            "Variation": variation,
            "Time_s": time,
            "Time_ms": time * 1000,
          
            "Drain_Current_A": current,
            "Drain_Current_nA": current_nA
        })

        extracted.append(trace_data)

    if len(extracted) == 0:
        raise ValueError(
            f"No valid transient data found in {filename}"
        )

    transient_data = pd.concat(
        extracted,
        ignore_index=True
    )

    return transient_data


# ============================================================
# 19. EXTRACT PPF DATA
# ============================================================

def extract_ppf_data(df, filename):

    variation = identify_variation(filename)

    print("\n============================================")
    print("PPF DATA EXTRACTION")
    print("============================================")
    print("Variation:", variation)

    ppf_records = []

    # --------------------------------------------------------
    # Search for A1, A2 and A2/A1 columns
    # --------------------------------------------------------

    columns_lower = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    a1_columns = [
        col for col in df.columns
        if str(col).strip().lower() == "a1"
    ]

    a2_columns = [
        col for col in df.columns
        if str(col).strip().lower() == "a2"
    ]

    ratio_columns = [
        col for col in df.columns
        if (
            "a2/a1" in str(col).lower()
            or "a2_a1" in str(col).lower()
            or "ppf" in str(col).lower()
            or "trr" in str(col).lower()
        )
    ]

    print("A1 columns:", a1_columns)
    print("A2 columns:", a2_columns)
    print("Ratio columns:", ratio_columns)

    # --------------------------------------------------------
    # Extract available PPF columns
    # --------------------------------------------------------

    max_length = max(
        [len(df[col]) for col in a1_columns + a2_columns + ratio_columns],
        default=0
    )

    for i in range(max_length):

        record = {
            "Variation": variation,
            "Point": i + 1
        }

        # A1
        if a1_columns:
            col = a1_columns[min(i, len(a1_columns) - 1)]

            value = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if i < len(value):
                record["A1_nA"] = value.iloc[i]

        # A2
        if a2_columns:
            col = a2_columns[min(i, len(a2_columns) - 1)]

            value = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if i < len(value):
                record["A2_nA"] = value.iloc[i]

        # A2/A1
        if ratio_columns:
            col = ratio_columns[min(i, len(ratio_columns) - 1)]

            value = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if i < len(value):
                record["A2_A1"] = value.iloc[i]

        ppf_records.append(record)

    ppf_data = pd.DataFrame(ppf_records)

    return ppf_data
# ============================================================
# SAVE EXTRACTED MODEL 1 DATA
# ============================================================

# 19. NORMALIZE DEVICE STATESFepo
# ============================================================

def normalize_device_states(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        raise ValueError(
            "No valid numerical device data found."
        )

    minimum = np.min(values)
    maximum = np.max(values)

    if np.isclose(
        maximum,
        minimum
    ):

        return np.ones_like(values)

    normalized = (
        values - minimum
    ) / (
        maximum - minimum
    )

    return normalized

# ============================================================
# 20. CREATE SIGNED DEVICE STATES
# ============================================================

def create_signed_device_states(
    normalized_states
):

    signed_states = (
        2.0 * normalized_states
    ) - 1.0

    signed_states = np.unique(
        signed_states
    )

    signed_states.sort()

    return signed_states

# ============================================================
# 21. DEVICE-AWARE WEIGHT QUANTIZATION
# ============================================================

def quantize_weights_to_device(
    trained_weights,
    device_states
):

    quantized_weights = []

    for weights in trained_weights:

        if weights.dtype.kind not in [
            'f',
            'c'
        ]:

            quantized_weights.append(
                weights.copy()
            )

            continue

        if weights.size == 0:

            quantized_weights.append(
                weights.copy()
            )

            continue

        weight_min = np.min(weights)
        weight_max = np.max(weights)

        if np.isclose(
            weight_max,
            weight_min
        ):

            quantized_weights.append(
                weights.copy()
            )

            continue

        # Normalize CNN weights to [0,1]
        normalized_weights = (
            weights - weight_min
        ) / (
            weight_max - weight_min
        )

        # Convert to [-1,1]
        signed_weights = (
            2.0 *
            normalized_weights
        ) - 1.0

        # Find nearest device state
        flat_weights = (
            signed_weights.ravel()
        )

        quantized_flat = np.empty_like(
            flat_weights
        )

        for i, weight in enumerate(
            flat_weights
        ):

            index = np.argmin(
                np.abs(
                    device_states - weight
                )
            )

            quantized_flat[i] = (
                device_states[index]
            )

        quantized_signed = (
            quantized_flat.reshape(
                signed_weights.shape
            )
        )

        # Convert back to original CNN weight range
        reconstructed = (
            (
                quantized_signed + 1.0
            ) / 2.0
        )

        reconstructed = (
            reconstructed *
            (
                weight_max -
                weight_min
            )
        ) + weight_min

        quantized_weights.append(
            reconstructed.astype(
                weights.dtype
            )
        )

    return quantized_weights

# ============================================================
# 22. DEVICE-AWARE CNN EVALUATION
# ============================================================

def evaluate_device_model(
    device_states,
    csv_name
):

    print("\n============================================")
    print("DEVICE-AWARE CNN")
    print("============================================")

    print(
        "CSV:",
        csv_name
    )

    print(
        "Number of device states:",
        len(device_states)
    )

    # Create fresh CNN
    device_model = build_cnn()

    # Start from exactly the same trained CNN
    device_model.set_weights(
        baseline_weights
    )

    # Apply device quantization
    device_weights = (
        quantize_weights_to_device(
            baseline_weights,
            device_states
        )
    )

    device_model.set_weights(
        device_weights
    )

    # Evaluate
    loss, accuracy = (
        device_model.evaluate(
            X_test,
            y_test_cat,
            verbose=0
        )
    )

    print(
        f"Device-aware accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # Prediction
    probabilities = (
        device_model.predict(
            X_test,
            batch_size=256,
            verbose=0
        )
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    true_labels = y_test

    precision = precision_score(
        true_labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall: {recall * 100:.2f}%"
    )

    print(
        f"F1-score: {f1 * 100:.2f}%"
    )

    return (
        device_model,
        loss,
        accuracy,
        precision,
        recall,
        f1,
        predictions
    )

# ============================================================
# 23. IDENTIFY DEVICE VARIATION
# ============================================================

def identify_variation(filename):

    name = filename.lower()

    if "delt" in name:

        return "Delta variation"

    elif "pw" in name:

        return "Pulse-width variation"

    elif "mag" in name:

        return "Magnitude variation"

    elif "equal" in name:

        return "Reference/Equal case"

    else:

        return "Other"

    # ============================================================
# 24. RESULT STORAGE
# ============================================================
# ============================================================
# 24. RESULT STORAGE
# ============================================================
# ============================================================
# 24. COMBINED CONFUSION MATRIX FOR ALL 8 CSV FILES
# ============================================================

all_results = []

# These two lists will store predictions from ALL 8 CSVs
all_true_labels = []
all_pred_labels = []

class_names = [
    chr(ord('A') + i)
    for i in range(num_classes)
]

print("\n============================================")
print("STARTING CASE 2 - ALL 8 CSV FILES")
print("============================================")

# ============================================================
# PROCESS ALL 8 CSV FILES
# ============================================================

for file_number, csv_file in enumerate(csv_files, start=1):

    csv_name = os.path.basename(csv_file)

    print("\n--------------------------------------------")
    print(f"Processing CSV {file_number} of {len(csv_files)}")
    print(f"File: {csv_name}")
    print("--------------------------------------------")

    # Read device CSV
    device_df = inspect_device_csv(csv_file)

    # Extract transient device data
    transient_data = extract_model1_data(
        device_df,
        csv_name
    )

    # Use only drain-current values as device states
    device_values = transient_data[
        "Drain_Current_nA"
    ].dropna().values

    if len(device_values) == 0:
        print("WARNING: No drain-current data found.")
        continue

    # Normalize device states
    normalized_states = normalize_device_states(
        device_values
    )

    # Create signed device states
    device_states = create_signed_device_states(
        normalized_states
    )

    # ========================================================
    # EVALUATE THIS CSV
    # ========================================================

    (
        device_model,
        device_loss,
        device_accuracy,
        precision,
        recall,
        f1,
        predictions
    ) = evaluate_device_model(
        device_states,
        csv_name
    )

    # ========================================================
    # ADD THIS CSV PREDICTIONS TO COMBINED DATA
    # ========================================================

    # True labels
    y_true_current = np.asarray(
        y_test
    ).reshape(-1)

    # Predicted labels
    y_pred_current = np.asarray(
        predictions
    ).reshape(-1)

    # ADD to combined lists
    all_true_labels.extend(
        y_true_current.tolist()
    )

    all_pred_labels.extend(
        y_pred_current.tolist()
    )

    print(
        f"Samples from this CSV: "
        f"{len(y_true_current)}"
    )

    print(
        f"Total samples collected: "
        f"{len(all_true_labels)}"
    )

    # ========================================================
    # SAVE INDIVIDUAL METRICS
    # ========================================================

    all_results.append({
        "CSV File": csv_name,
        "Accuracy (%)": device_accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1 Score (%)": f1 * 100
    })
# ============================================================
# 26. ONE COMBINED CONFUSION MATRIX + COMBINED ACCURACY
# ============================================================

print("\n============================================")
print("Model 1 CONFUSION MATRIX")
print("============================================")

all_true_labels = np.asarray(
    all_true_labels,
    dtype=int
)

all_pred_labels = np.asarray(
    all_pred_labels,
    dtype=int
)

# Check
if len(all_true_labels) != len(all_pred_labels):
    raise ValueError(
        "True labels and predicted labels have different lengths."
    )

# Create ONE confusion matrix from ALL 8 CSVs
cm_combined = confusion_matrix(
    all_true_labels,
    all_pred_labels,
    labels=np.arange(num_classes)
)

# Calculate combined accuracy
correct_predictions = np.trace(cm_combined)

total_predictions = np.sum(cm_combined)

combined_accuracy = (
    correct_predictions / total_predictions
)

print("\n============================================")
print("COMBINED RESULTS - ALL 8 CSVs")
print("============================================")

print(
    "Correct predictions:",
    correct_predictions
)

print(
    "Total predictions:",
    total_predictions
)

print(
    f"Combined Accuracy: "
    f"{combined_accuracy * 100:.2f}%"
)

# ============================================================
# PLOT ONE COMBINED CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(14, 12)
)

sns.heatmap(
    cm_combined,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Letter")

plt.ylabel("True Letter")

plt.title(
    f"Combined Confusion Matrix - Case 2 - "
    f"All 8 ZrO₂/MgO Device Variations\n"
    f"Combined Accuracy = "
    f"{combined_accuracy * 100:.2f}%"
)

plt.tight_layout()

# ============================================================
# SAVE COMBINED CONFUSION MATRIX
# ============================================================

output_path = os.path.join(
    RESULT_FOLDER,
    "Case2_Combined_Confusion_Matrix_All_8_CSVs.png"
)

plt.savefig(
    output_path,
    dpi=600,
    bbox_inches="tight"
)

print("\n============================================")
print("COMBINED CONFUSION MATRIX SAVED")
print("============================================")

print(
    output_path
)

plt.show()

plt.close()


# ============================================================
# SAVE RESULTS OF ALL 8 CSVs
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = os.path.join(
    RESULT_FOLDER,
    "CASE2_RESULTS.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n============================================")
print("CASE 2 COMPLETED")
print("============================================")

print(
    "Metrics saved at:",
    results_path
)

print("\nIndividual CSV Results:")
print(results_df)
