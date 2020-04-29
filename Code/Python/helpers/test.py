import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        #The string to be sent should be in the format: label + white space + x + white space + y
        #For example when retreiving shampoo: 2 0 1.3        
	#For example when retreiving shampoo: 2 -1.3 1.3
        str1 = input("Enter label and coordinates: ")
        s.sendall(str1.encode())

