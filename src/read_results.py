import sys
res_file = sys.argv[1]
results_file = res_file
line_number = 0
all_results = []
current_result = []
for line in open(results_file, "r"):
    line_number += 1
    line = line[:-1]
    current_result.append(line)
    if line_number%33 == 0:
        all_results.append(current_result[:])
        current_result = []

for result in all_results:
    params = "\t".join((result[0].split(" ")))
    men_result = result[8]
    rw_result = result[16]
    simlex_result = result[24]
    ws353_result = result[32]
    print(params + "\t" + men_result + "\t" + rw_result + "\t" + simlex_result + "\t" + ws353_result )
    
