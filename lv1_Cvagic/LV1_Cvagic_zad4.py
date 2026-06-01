ime = input("Ime datoteke: ")

fhand = open(ime)

suma = 0
brojac = 0

for line in fhand:
    if line.startswith("X-DSPAM-Confidence:"):
        broj = float(line.split(":")[1])
        suma += broj
        brojac += 1

fhand.close()

print("Average X-DSPAM-Confidence:", suma / brojac)
