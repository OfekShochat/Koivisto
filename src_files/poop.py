import struct

f = open("hey.bin", "rb").read()
d = "=??i??h?ii??i?ii?i";
a = 0
for i in d:
    if i == "?":
        a += 1
    elif i == "i":
        a += 4
    elif i == "h":
        a += 2
print(a)
a = struct.unpack(d, f[:39])
print(a)
