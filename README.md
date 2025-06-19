# Real-time Gesture Recognition System

An interactive, deep-learning-based system for real-time hand gesture recognition and control, featuring on-the-fly gesture registration and automated background retraining.

---

## Key Features

- **Real-time Recognition**: Live gesture classification via webcam.
- **On-the-Fly Gesture Registration**: Add new gestures and map them to existing actions without restarting the application.
- **Automated Background Retraining**: The model automatically retrains in a separate process after a new gesture is added.
- **Automatic Model Reloading**: The application detects and loads the newly trained model seamlessly.
- **Firebase Integration**: Updates device status and logs actions to a Firebase Realtime Database.
- **JSON-based Configuration**: Easily manage gesture classes in `gestures.json`.

---

## Environment

Tested and developed with:

- **OS**: Windows 11 / macOS
- **Python**: 3.10+

---

## Project Structure

```
.
├── inference_final.py       # Main application script
├── train.py                 # Model training script
├── config.py                # System configuration and constants
├── firebase_utils.py        # Firebase utility functions
├── gestures.json            # Gesture class definitions
│
├── dataset.py               # Custom PyTorch Dataset class
├── model.py                 # CNN-BiGRU model architecture
│
├── train_data/              # Directory for training data
└── model/                   # Directory for the saved model
```

---

## Setup

### 1. Clone the repository

### 2. Install dependencies

```bash
pip install torch opencv-python mediapipe numpy pandas tqdm firebase-admin
```

### 3. Firebase Credentials

---

## Usage

### 1. Run the Application

```bash
python inference_final.py
```

### 2. Execute an Action

Gesture flow:

```text
Trigger_Action → Yes → [Action Gesture] → Yes
```

### 3. Register a New Gesture

#### Start Registration

```text
Trigger_Register → Yes
```

#### Select Target Action

When prompted:

> Map to which action?

- Show the original gesture for the action (e.g., 'Turn on Light').

#### Confirm and Record

- Confirm with **Yes**
- 5-second countdown begins
- Perform your **new custom gesture** for 15 seconds

#### Save and Train

- Confirm save with **Yes**
- When asked:

> Start training now?

- Confirm with **Yes**
- Training runs in the background and reloads the model

---

## Configuration

- `gestures.json`: Defines gesture names and class IDs
- `config.py`: System constants (e.g. paths, trigger gesture names)

---

## Default Gesture Classes

The following gestures are pre-defined in `gestures.json`:

| ID  | Gesture Name       | Description                         | Gesture Motion (Example)                                     |
|-----|--------------------|-------------------------------------|--------------------------------------------------------------|
| 0   | Turn on Light      | Activates the light device          | Stretch hand with fingers slightly bent and shake (L shape)  |
| 1   | Turn off Light     | Deactivates the light device        | Rotate a closed fist upward                                  |
| 2   | Turn on Fan        | Activates the fan                   | Wave an open hand toward the camera like fanning             |
| 3   | Turn off Fan       | Deactivates the fan                 | Point with index finger and rotate toward the camera         |
| 4   | Turn on Music      | Starts music playback               | Shake your hand in the “call me” sign                        |
| 5   | Turn off Music     | Stops music playback                | Show palm and wave side to side                              |
| 6   | Curtain Open       | Opens the curtain                   | Open palm and repeatedly clench into a fist                  |
| 7   | Curtain Close      | Closes the curtain                  | Make an 'OK' sign and shake                                  |
| 8   | Yes                | Confirms a prompt                   | Thumbs up and shake                                          |
| 9   | No                 | Cancels or rejects a prompt         | Thumbs down and shake                                        |
| 10  | Trigger_Action     | Initiates gesture-based action mode | make '1' and shake                                           |
| 11  | Trigger_Register   | Initiates new gesture registration  | make '2' and shake                                           |

These classes can be customized or extended via `gestures.json`.

---

## Model

The system uses a **CNN-BiGRU** architecture for:

- Spatial feature extraction (CNN)
- Temporal sequence modeling (BiGRU)