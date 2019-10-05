import re

templates_dict = {}
with open("zh_merge_result_zh_en.txt", encoding="utf-8") as q_t_reader:
	with open("new_zh_en_triple.txt", encoding="utf-8") as triples_reader:
		with open("zh_zh_en.txt", "w", encoding="utf-8") as writer:
			for line in q_t_reader.readlines():
				line = line.strip().split('\t')
				rel_combination = re.findall(r'''(?<=['"])[^',]+(?=['"])''', line[0])
				rel_comb_str = rel_combination[0] + "," + rel_combination[1]
				template = re.findall(r'''(?<=")[^",]+(?=")''', line[1])
				templates_dict[rel_comb_str] = template

			for line in triples_reader.readlines():
				line = line.strip()
				triple_combination = re.findall(r"(?<=/)[^/]+(?=>)", line)
				rel_comb_str = triple_combination[1] + "," + triple_combination[4]
				question = ""
				if rel_comb_str in templates_dict.keys():
					template = templates_dict[rel_comb_str]
					for temp in template:
						t = temp.split("entity")
						try:
							question = t[0] + triple_combination[0] + t[1]
						except:
							print(t)
						else:
							data = "{}@@@{}@@@{}#{}#{}#{}#{}#{}".format(question, triple_combination[5], triple_combination[0],
						                                            triple_combination[1], triple_combination[2],
						                                            triple_combination[3], triple_combination[4],
						                                            triple_combination[5])
							writer.write(data + "\n")