"""
ws_test.py  –  quick WebSocket end-to-end smoke test
Run: python ws_test.py
"""
import asyncio
import base64
import json

import cv2
import numpy as np
import websockets


async def test():
    # Create a small dummy frame and JPEG-encode it
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:180, 80:240, :] = 200   # white-ish rectangle
    _, buf = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buf).decode()

    uri = 'ws://localhost:8000/ws/pose'
    async with websockets.connect(uri) as ws:

        # Test 1: valid frame, valid exercise (blank frame → no pose detected)
        await ws.send(json.dumps({'exercise': 'Bicep Curl', 'frame': b64}))
        resp = json.loads(await ws.recv())
        print('Test 1 – blank frame / Bicep Curl:')
        print(json.dumps(resp, indent=2))

        # Test 2: missing exercise field
        await ws.send(json.dumps({'frame': b64}))
        err = json.loads(await ws.recv())
        print('\nTest 2 – missing exercise field:')
        print(json.dumps(err, indent=2))

        # Test 3: unknown exercise name
        await ws.send(json.dumps({'exercise': 'FlipFlop', 'frame': b64}))
        err = json.loads(await ws.recv())
        print('\nTest 3 – unknown exercise name:')
        print(json.dumps(err, indent=2))

        # Test 4: data-URL prefix (browser-style)
        b64_url = f'data:image/jpeg;base64,{b64}'
        await ws.send(json.dumps({'exercise': 'Squats', 'frame': b64_url}))
        resp = json.loads(await ws.recv())
        print('\nTest 4 – data-URL prefix / Squats:')
        print(json.dumps(resp, indent=2))

asyncio.run(test())
