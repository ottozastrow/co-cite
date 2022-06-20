str = "1vet.app.49(c), 55 (1990)" # -> 1vet.app.49(c), 55 

if str[-1] == ")" and str[-2].isnumeric():
    print("yes")
    print(str.rsplit("(", 1)[0])