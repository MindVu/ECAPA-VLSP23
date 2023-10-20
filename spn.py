substitution = [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7]
permutation = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]

key = ['0011101010010100',
       '1010100101001101',
       '1001010011010110',
       '0100110101100011',
       '1101011000111111']


def decimalToBinary(decimal, bits):
    binary_str = bin(decimal)[2:]
    while len(binary_str) < bits:
        binary_str = '0' + binary_str

    return binary_str


def nameToBinary(name):
    str = ''
    for c in name:
        str += decimalToBinary(ord(c) - ord('a') - 1, 5)
    print(formattedString(str))
    return str[:16]


def formattedString(str):
    formatted_str = ''
    for i in range(0, len(str), 4):
        formatted_str += str[i: i + 4]
        formatted_str += ' '
    return formatted_str


plain_text = nameToBinary("minh")
# plain_text = '011010'
formattedString(plain_text)

def xor_block(plain_text, key):
    u = ''
    for j in range(0, 16, 4):
        num1 = int(plain_text[j: j + 4], 2)
        num2 = int(key[j: j + 4], 2)
        xor_res = num1 ^ num2
        u += decimalToBinary(xor_res, 4)
    return u

def substitute_string(s, substitution):
    v = ''
    for j in range(0, 16, 4):
        v += decimalToBinary(substitution[int(s[j: j + 4], 2)], 4)
    return v
def permute_string(s, permutation):
    w = ''
    for idx in permutation:
        w += v[idx - 1]
    return w
    
for i in range(3):
    print('w' + str(i) + ': ' + formattedString(plain_text))
    print('k' + str(i + 1) + ': ' + formattedString(key[i]))
    u = xor_block(plain_text, key[i])
    v = substitute_string(u, substitution)

    print('u' + str(i + 1) + ': ' + formattedString(u))
    print('v' + str(i + 1) + ': ' + formattedString(v))

    
    plain_text = permute_string(v, permutation)
    print('-' * 30)

print('w4: ' + formattedString(plain_text))
print('k4: ' + formattedString(key[i]))
u = xor_block(plain_text, key[3])
v = substitute_string(u, substitution)

print('u4: ' + formattedString(u))
print('v4' + ': ' + formattedString(v))
print('k5: ' + formattedString(key[i]))

plain_text = v
u = xor_block(plain_text, key[4])

print('y ' + ': ' + formattedString(u))