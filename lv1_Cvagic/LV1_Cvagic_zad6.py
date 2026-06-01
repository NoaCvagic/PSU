fhand = open("SMSSpamCollection.txt", encoding="utf-8")

ham_broj = 0
spam_broj = 0

ham_rijeci = 0
spam_rijeci = 0

for line in fhand:
    line = line.strip()

    dijelovi = line.split()

    tip = dijelovi[0]
    poruka = dijelovi[1:]

    if tip == "ham":
        ham_broj += 1
        ham_rijeci += len(poruka)

    elif tip == "spam":
        spam_broj += 1
        spam_rijeci += len(poruka)

fhand.close()

print("Prosjek HAM:", ham_rijeci / ham_broj)
print("Prosjek SPAM:", spam_rijeci / spam_broj)

fhand = open("SMSSpamCollection.txt", encoding="utf-8")

brojac = 0

for line in fhand:
    line = line.strip()

    if line.startswith("spam"):
        poruka = line[5:]

        if poruka.endswith("!"):
            brojac += 1

fhand.close()

print("Broj spam poruka koje zavrsavaju usklicnikom:", brojac)