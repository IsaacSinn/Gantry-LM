# Gantry LM
 language models to control a XY gantry with natural language commands

## Vocab

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

Output tokens: 
```
<BOS>: Beginning of sequence
<EOS>: End of sequence

ROTATE
MOVE
SPEED
0-9
<space>
```

## Gantry setup and motor specs 

A 50cm x 50cm 2d plane, top left of the plate is (0,0) and bottom right is (50,50). The motor that controls the X axis or th horizontal plane is ```motor_id``` 0 and the motor that controls the Y axis, the vertical plane is ```motor_id``` 1. Clockwise on X axis / ```motor_id``` 0 moves the gantry to the right, clockwise on the Y axis / ```motor_id``` 1 moves the gantry down.

Each motor has a shaft circumference of 2 cm. Therefore every 360 degree is 2 cm moved

## Example input output

Move the header from top left to bottom right:
```
MOVE 0 1 50
MOVE 1 1 50
```