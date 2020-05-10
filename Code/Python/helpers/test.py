import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
    # The string to be sent should be in the format: label + white space + x + white space + y
        try:
            str1 = input("Enter label and coordinates: ")
            s.sendall(str1.encode())
            data = s.recv(1024)
            # The directions being received are Map directions relative to the X and Y of the Map
            # Should be converted into rover actions depending on its orientation (rotate, move etc)
            print(data.decode())
        except:
            break

