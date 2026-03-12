from statistics import mean
print("Unesi brojeve i na kraju unesi Done")
brojevi = []
while True:
    unos = input()
    if unos.lower() == "done":
        break
    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Nije unesen broj")
if brojevi:
    print("Najveci broj je: ", max(brojevi))
    print("Najmanji broj je: ", min(brojevi))
    print("Srednja vrijednost brojeva je: ", mean(brojevi))
else:
    print("Nisu uneseni brojevi")