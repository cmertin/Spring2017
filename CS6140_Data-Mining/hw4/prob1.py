from collections import Counter
from countminsketch import CountMinSketch

def ReadFile(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    return lines[0]

def Misra_Gries(stream, k):
    counts = Counter()
    for e in stream:
        if e in counts or len(counts) < k:
            counts[e] += 1
        else:
            for c in list(counts.keys()):
                counts[c] = counts[c] - 1
                if counts[c] == 0:
                    del counts[c]
    return counts.most_common(k)

# k = table size
# h = hash functions
def CountMin_Sketch(stream, k, h):
    sketch = CountMinSketch(k, h)
    for e in stream:
        sketch.add(e)

    return sketch

f1 = "S1.txt"
f2 = "S2.txt"
k = 9
t = 10
h = 5

print(f1)
str_f1 = ReadFile(f1)
mg = Misra_Gries(str_f1, k)
print(mg)
cm = CountMin_Sketch(str_f1, t, h)
print("a: " + str(cm.query('a')))
print("b: " + str(cm.query('b')))
print("c: " + str(cm.query('c')))

print('\n\n')
print(f2)
str_f2 = ReadFile(f2)
mg = Misra_Gries(str_f2, k)
print(mg)
cm = CountMin_Sketch(str_f2, t, h)
print("a: " + str(cm.query('a')))
print("b: " + str(cm.query('b')))
print("c: " + str(cm.query('c')))
