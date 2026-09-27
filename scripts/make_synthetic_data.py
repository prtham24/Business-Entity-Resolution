"""Generate a small synthetic dataset in the challenge format (the real data cannot be shared).

Creates <out>/train/{train_source1,2,3,train_ground_truth}.tsv and
        <out>/test/{test_source1,2,3,test_ground_truth}.tsv
Train covers US and India; test adds France (an unseen country), mirroring the real challenge.
Noise: abbreviations, legal-suffix changes, typos, casing, DBA names, dropped address parts,
landmark prefixes, word swaps; plus same-name-different-address hard negatives and distractors.

Usage: python scripts/make_synthetic_data.py data_synth [--train-s1 4000] [--test-s1 3000]
"""
import argparse
import os
import random
import re

W = ["alpha", "bright", "crystal", "delta", "eagle", "falcon", "global", "horizon", "indigo", "jupiter",
     "krishna", "lotus", "metro", "nova", "orbit", "prime", "quantum", "royal", "summit", "titan", "unity",
     "vertex", "wisdom", "zenith", "ganga", "shakti", "sai", "balaji", "laxmi", "shree", "sunrise", "silver",
     "golden", "green", "blue", "ocean", "river", "star"]
IND = ["traders", "enterprises", "industries", "solutions", "technologies", "services", "foods", "motors",
       "pharma", "textiles", "logistics", "builders", "electricals", "hardware", "jewellers", "exports",
       "consultants", "systems"]
FR_W = ["boulangerie", "dupont", "lefevre", "moreau", "atelier", "garage", "maison", "cafe", "epicerie",
        "bernard", "martin", "durand", "petit", "renard", "lumiere", "etoile", "soleil", "riviere"]
LEGAL = {"US": ["Inc", "LLC", "Corp", "Co", "Corporation", "Incorporated"],
         "India": ["Pvt Ltd", "Private Limited", "Ltd", "LLP", "Limited"],
         "France": ["SARL", "SAS", "SA", "EURL", "Société Anonyme"]}
CITY = {"US": [("Austin", "TX"), ("Denver", "CO"), ("Seattle", "WA"), ("Boston", "MA"), ("Miami", "FL")],
        "India": [("Lucknow", "Uttar Pradesh"), ("Pune", "Maharashtra"), ("Jaipur", "Rajasthan"),
                  ("Chennai", "Tamil Nadu"), ("Bengaluru", "Karnataka")],
        "France": [("Paris", ""), ("Lyon", ""), ("Marseille", ""), ("Toulouse", ""), ("Nantes", "")]}
STREETS = {"US": ["Main Street", "Oak Avenue", "Pine Road", "Maple Drive", "Lake Boulevard", "Park Lane"],
           "India": ["MG Road", "Station Road", "Gandhi Nagar", "Civil Lines", "Sector 12", "Hazratganj Marg"],
           "France": ["Rue de la Paix", "Avenue Victor Hugo", "Boulevard Saint-Michel", "Rue des Lilas"]}
SWAPS = [("Private Limited", "Pvt Ltd"), ("Pvt Ltd", "Private Limited"), ("Corporation", "Corp"),
         ("Corp", "Corporation"), ("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Boulevard", "Bd"),
         ("Limited", "Ltd"), ("Société Anonyme", "SA"), ("Uttar Pradesh", "UP"), ("Rue", "R."),
         ("Incorporated", "Inc")]


def base_entity(country):
    if country == "France":
        name = " ".join(random.sample(FR_W, 2)).title() + " " + random.choice(LEGAL[country])
    else:
        name = (" ".join(random.sample(W, random.choice([1, 2]))).title() + " "
                + random.choice(IND).title() + " " + random.choice(LEGAL[country]))
    city, state = random.choice(CITY[country])
    num = random.randint(1, 999)
    street = random.choice(STREETS[country])
    if country == "US":
        addr = f"{num} {street}, {city}, {state} {random.randint(10000, 99999)}"
    elif country == "India":
        addr = f"{num}, {street}, {city}, {state} {random.randint(110000, 859999)}"
    else:
        addr = f"{num} {street}, {random.randint(10000, 95999)} {city}"
    return name, addr


def typo(s):
    if len(s) < 5:
        return s
    k = random.randint(1, len(s) - 2)
    r = random.random()
    if r < 0.33:
        return s[:k] + s[k + 1:]
    if r < 0.66:
        return s[:k] + s[k + 1] + s[k] + s[k + 2:]
    return s[:k] + random.choice("aeioustrn") + s[k:]


def swap_word(text, a, b):
    return re.sub(r"\b" + re.escape(a) + r"\b", b, text)


def noisy(name, addr):
    for a, b in SWAPS:
        if random.random() < 0.5:
            name = swap_word(name, a, b)
        if random.random() < 0.5:
            addr = swap_word(addr, a, b)
    if random.random() < 0.5:
        name = name.replace(" and ", " & ")
    if random.random() < 0.3:
        name = typo(name)
    if random.random() < 0.3:
        addr = typo(addr)
    if random.random() < 0.2:
        name = name.upper()
    if random.random() < 0.15:
        name = name + " dba " + random.choice(W).title() + " Store"
    if random.random() < 0.2:
        parts = addr.split(", ")
        if len(parts) > 2:
            addr = ", ".join(parts[:-1])
    if random.random() < 0.15:
        addr = "Near SBI ATM, " + addr
    if random.random() < 0.1:
        t = name.split()
        if len(t) > 2:
            t[0], t[1] = t[1], t[0]
            name = " ".join(t)
    return name, addr


def make(split, countries, n_s1, out):
    os.makedirs(out, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    c2 = c3 = 0
    for k in range(n_s1):
        country = random.choice(countries)
        name, addr = base_entity(country)
        sid = f"S1-{k:05d}"
        s1.append((sid, name, addr, country))
        matches = []
        for target in ("s2", "s3"):
            for _ in range(random.choices([0, 1, 2, 3], [0.35, 0.45, 0.15, 0.05])[0]):
                nm, ad = noisy(name, addr)
                if target == "s2":
                    c2 += 1
                    rid = f"S2-{c2:05d}"
                    s2.append((rid, nm, ad, country))
                else:
                    c3 += 1
                    rid = f"S3-{c3:05d}"
                    s3.append((rid, nm, ad, country))
                matches.append(rid)
        gt.append((sid, ",".join(matches)))
        if random.random() < 0.15:          # hard negative: same name, different business location
            _, addr2 = base_entity(country)
            c2 += 1
            s2.append((f"S2-{c2:05d}", noisy(name, addr2)[0], addr2, country))
    for _ in range(n_s1 // 3):             # distractors with no Source-1 partner
        country = random.choice(countries)
        nm, ad = base_entity(country)
        c3 += 1
        s3.append((f"S3-{c3:05d}", nm, ad, country))
    random.shuffle(s2)
    random.shuffle(s3)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for fn, rows in ((f"{split}_source1.tsv", s1), (f"{split}_source2.tsv", s2), (f"{split}_source3.tsv", s3)):
        with open(os.path.join(out, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            for r in rows:
                f.write("\t".join(r) + "\n")
    with open(os.path.join(out, f"{split}_ground_truth.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for r in gt:
            f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="data_synth")
    ap.add_argument("--train-s1", type=int, default=4000)
    ap.add_argument("--test-s1", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    make("train", ["US", "India"], a.train_s1, os.path.join(a.out, "train"))
    make("test", ["US", "India", "France"], a.test_s1, os.path.join(a.out, "test"))
    print(f"synthetic data written to {a.out}/ (test ground truth included for local scoring)")
