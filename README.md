# Gantry LM

Fine-tuned StarCoder2 3B language model to control an XY gantry with natural language commands by generating G-code.

## Goal and Gantry Setup

The system operates on a 50cm x 50cm 2D plane:
- Top left of the plate is (0,0) and bottom right is (50,50)
- `motor_id` 0 controls the X axis (horizontal plane)
- `motor_id` 1 controls the Y axis (vertical plane)
- Clockwise on X axis moves the gantry to the right
- Clockwise on Y axis moves the gantry down

Each motor has a shaft circumference of 2 cm, meaning every 360° rotation moves the gantry 2 cm. The goal is to move the gantry using natural language commands.

## Approach Evolution

### Initial Approach: Self-defined Vocabulary (not used)

#### Output Tokens
```
<EOS>: End of sequence
<EOI>: End of instruction (marks the end of input instruction)
<SC>: Start of concurrent execution
<EC>: End of concurrent execution
ROTATE: tool call
MOVE: tool call
SPEED: tool call
0-9: numbers 0 to 9
" ": space
".": decimal point
```

#### Available Tool Calls
```
ROTATE <motor_id> <direction> <degree> 
- direction: 0 (CCW / Counter Clockwise), 1 (CW / Clockwise)
- degree: positive float

MOVE <motor_id> <direction> <distance>
- distance: in cm

SPEED <motor_id> <direction> <speed> <duration>
- duration in seconds
- speed in degree per second
```

#### Example Data
```
Move the header from top left to bottom right one servo at a time. <EOI> MOVE 0 1 50 MOVE 1 1 50
```

```
Rotate the first servo 90 degree clockwise and second servo 90 degree counterclockwise at the same time <EOI> <SC> ROTATE 0 1 90 ROTATE 1 0 90 <EC>
```

```
move in a square of length 10 cm in the middle of the board, the gantry system is currently in the center (25,25). <EOI> <SC> MOVE 0 0 5 MOVE 1 0 5 <EC> MOVE 0 1 10 MOVE 1 1 10 MOVE 0 0 10 MOVE 1 0 10
```
Explanation: move from the middle to the top left of the square in a diagonal. And then draw the square one move at a time.

```
Move the motor 0 at a speed of 90 degree per second clockwise <EOI> SPEED 0 1 90
```

#### Issues with Self-defined Vocabulary
- Hard to find data to train on
- Using LLM models to generate G-code (distillation) is challenging as LLMs are not trained on data with our custom vocabulary

### Current Approach: G-code (used)

G-code is a standard language used to control CNC machines and 3D printers, specifically for controlling motor movements.

#### Example G-code
```
# draw a circle with radius 10 cm from the center of the board
G1 X0 Y0 F1000 # move to the center of the board
G2 X10 Y0 I10 J0 # draw a circle with radius 10 cm from the center of the board
```

#### Example Data Format (JSONL)
```
{
    "prompt": "move to the center of the board and draw a circle with radius 10 cm from the center of the board",
    "completion": "G1 X0 Y0 F1000\nG2 X10 Y0 I10 J0<|endoftext|>"
}
```

#### Sample Data For Running
train_synthetic_1100.jsonl and dev_100.jsonl are generated using the data/stacks_gcode_jsonl.py script.

## Models

### Baseline RNN
- Token level: character level except special characters
- Input vector: one-hot vectors `1 x |Vocab|`
- RNN architecture: 1 hidden layer with 64 nodes
- Technique: next token prediction

### Transformer

Some imports may depend on directory structure.
Run:
```
python model/transformer/train_test_transformer.py
```

### StarCoder2 3B

Run:
1. Install requirements
```
pip install -r requirements.txt
```
2. Install cudatoolkit, torch according to your hardware
3. Training: Run changing according to your hardware:
```
accelerate launch model\starcoder-3b\train_starcoder.py \
--model_id "bigcode/starcoder2-3b" \
--dataset_name "json" \
--dataset_path "data/train_synthetic_1100.jsonl" \
--dataset_text_field "completion" \
--prompt_field "prompt" \
--max_seq_length 4096 \
--max_steps 1000 \
--micro_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 2e-4 \
--warmup_steps 100 \
--num_proc 8

```
4. Inference:
Change the checkpoint path in the script
```
python model/starcoder-3b/starcoder_gcode_inference.py
```
