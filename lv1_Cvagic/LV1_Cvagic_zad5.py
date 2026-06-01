fhand = open("song.txt")

rijeci = {}

for line in fhand:
    words = line.split()

    for word in words:
        rijeci[word] = rijeci.get(word, 0) + 1

fhand.close()

jednom = []

for rijec, broj in rijeci.items():
    if broj == 1:
        jednom.append(rijec)

print("Broj rijeci koje se pojavljuju jednom:", len(jednom))
print("Rijeci:")

for rijec in jednom:
    print(rijec)