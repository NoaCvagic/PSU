print("Unesi broj izmedu 0.0 i 1.0: ")
try:
    broj = float(input())    

    if broj < 0.0 or broj > 1.0:
        print("Broj nije izmedu 0.0 i 1.0")
    elif broj >= 0.9:
        print("A")
    elif broj >= 0.8:
        print("B")
    elif broj >= 0.7:
        print("C")
    elif broj >= 0.6:
        print("D")
    else:
        print("F")


except:
    print("Nije unesen broj")
  