# Gantry LM
 language models to control a XY gantry with natural language commands

## Vocab

Available Tool calls: 
```
ROTATE <motor_id> <direction> <degree> 
- direction: 0 (CCW), 1 (CW)
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
```