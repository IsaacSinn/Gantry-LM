# Gantry LM
 language models to control a XY gantry with natural language commands

## Vocab

Available Tool calls: 
```
ROTATE <motor_id> <degree> <direction>
- direction: 0 (CCW), 1 (CW)
- degree: positive float

MOVE <motor_id> <distance>
- distance: in cm

SPEED <motor_id> <speed> <direction> <duration>
- duraction in seconds
- speed in degree per second
```

Output tokens: 
```
<BOS>: Beginning of sequence
<EOS>: End of sequence

ROTATE
MOVE
SPEED

```