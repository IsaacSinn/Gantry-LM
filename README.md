# Gantry LM
 language models to control a XY gantry with natural language commands

## Vocab
Output tokens: 
```
<EOS>: End of sequence
<EOI>: End of instruction. This marks the end of the input instruction for the LM
<SC>: Start of concurrent. Some instructions are required to be executed concurrently. For example, drawing a circle on the gantry
<EC>: End of concurrent. marks end of cocurrent execution

ROTATE: tool call
MOVE: tool call
SPEED: tool call
0-9: numbers 0 to 9
" ": space (quotation not included in output token)
".": decimal point (quotation not included in output token)
```

Available Tool calls: 
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

## Gantry setup and motor specs 

A 50cm x 50cm 2d plane, top left of the plate is (0,0) and bottom right is (50,50). The motor that controls the X axis or th horizontal plane is ```motor_id``` 0 and the motor that controls the Y axis, the vertical plane is ```motor_id``` 1. Clockwise on X axis / ```motor_id``` 0 moves the gantry to the right, clockwise on the Y axis / ```motor_id``` 1 moves the gantry down.

Each motor has a shaft circumference of 2 cm. Therefore every 360 degree is 2 cm moved

## Example data
```
Move the header from top left to bottom right one servo at a time. <EOI> MOVE 0 1 50 MOVE 1 1 50
```

```
Rotate the first servo 90 degree clockwise and second servo 90 degree counterclockwise at the same time <EOI> <SC> ROTATE 0 1 90 ROTATE 1 0 90 <EC>
```

```
move in a square of length 10 cm in the middle of the board, the gantry system is currently in the center (25,25). <EOI> <SC> MOVE 0 0 5 MOVE 1 0 5 <EC> MOVE 0 1 10 MOVE 1 1 10 MOVE 0 0 10 MOVE 1 0 10
```
Explanation: move from the middle to the top left of the square in a diaganol. And then draw the square one move at a time

```
Move the motor 0 at a speed of 90 degree per second clockwise <EOI> SPEED 0 1 90
```


# Models

## Baseline RNN

token level: character level except special characters

input vector: one hot vectors ```1 x |Vocab|```

RNN architecture: 1 hidden layer with 64 nodes

technique: next token prediction