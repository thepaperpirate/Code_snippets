import socket , time


HOST = "hasx0016dem"
NOTIFICATION_PORT = 1080 #select a port that you know is open
port : (int) = 0


notification_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
notification_socket.connect((HOST,NOTIFICATION_PORT))
print(f"connected to {HOST}:{NOTIFICATION_PORT}")


while True:

    notification_socket.sendall(str("READY").encode("utf-8"))
    next_port= notification_socket.recv(1024)
    
    if not next_port:
        print("Notification server closed connection")
        break

    print("next port received")

    try:
        port = int(next_port)
    except ValueError:
        print("received invalid port number")
        continue 

    print(f"attempting connecting on port: {port}")   
    Client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    Client_socket.settimeout(2)

    try:
        Client_socket.connect((HOST,port))
        msg = Client_socket.recv(1024).decode("utf-8")
        if msg != "":
            with open("port_log.txt", "a" , encoding="utf-8") as f:
                f.write(f"{msg}_: under {port} \n")
        print(msg)
    except OSError as error:
        print(f"port {port} verbindung fehlgeschlagen: {error}")
    



    #print(f"attempting connection on port {next_port}")
    #timeout = time.monotonic() + 1
    #msg = ""
    #while time.monotonic() < timeout and msg == "":
   
    #    try:
    #        Client_socket.connect(("hasx0016dem",port))
    #        msg = Client_socket.recv(1024)
    #        msg = int(msg.decode("utf-8"))
    #        if msg != "":
    #            with open("port_log.txt", "a" , encoding="utf-8") as f:
    #                f.write(f"{msg}_: under {port}")
    #           print(msg)
    #    except OSError as error:
    #        print(f"port {port} verbindung fehlgeschlagen: {error}")