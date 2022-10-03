# %%
my_name = "Yuan_Chi_Lin"
my_ID = "0811074"

seraching_range = [200, 9900]
#print(seraching_range)

# %%
prime = [1, 2]

for test_number in range(3, seraching_range[1]+1, 2):
    is_prime = True

    for prime_number in prime:
        if((test_number % prime_number) == 0 and prime_number != 1):
            is_prime = False
            break

    if(is_prime):
        prime.append(test_number)

for i in range(0, len(prime)):

    if(prime[0] < seraching_range[0]):
        del prime[0]
    else:
        break

# print(prime)
    

# %%
file_path = "./" + my_name + "_prime_found.txt"
#print(file_path)

file = open(file_path, 'w')

count = 0

for i in range(1, (len(prime) + 1)):
    count += 1
    
    file.write(f"{prime[-1*i]} ")

    if(count == 6):
        file.write("\n")
        count = 0


file.close()

# %%
input_file = open(file_path, 'r')

data = input_file.read()
data = data.split()

for i in range(0, len(data)):
    data[i] = int(data[i])

# %%
target_range =  [3000, 6000]

target_count = 0

for i in data:
    if(i >= target_range[0] and i <= target_range[1]):
        target_count +=1
    elif(i < target_range[0]):
        break

print(f"I,{my_name},{my_ID},found {target_count} prime numbers between {target_range[0]} and {target_range[1]}")

input_file.close()


