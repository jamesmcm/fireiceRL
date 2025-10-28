from fireicerl.bridge import FCEUXBridge


bridge = FCEUXBridge()
print("handshake:", bridge._request({"cmd": "handshake", "client": "probe"}))
print("reset keys:", bridge.reset().keys())
