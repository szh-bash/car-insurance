import socket
from config import server, modelPath, modelSavePath

ip_port = ('127.0.0.1', server)
s = socket.socket()
s.connect(ip_port)
print('connected')
# s.sendall(modelPath.encode())
s.sendall((modelSavePath+"_10.tar").encode())
# s.sendall('exit'.encode())
s.close()
