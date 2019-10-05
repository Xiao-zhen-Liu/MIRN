import re

a_dict = {}
en_triples = []
zh_triples = []
with open("new_en_zh_triple.txt", encoding="utf-8") as reader_1, open("new_zh_en_triple.txt",
                                                                      encoding="utf-8") as reader_2:
	for line in reader_1.readlines():
		line = line.strip()
		triple_combination = re.findall(r"(?<=/)[^/]+(?=>)", line)
		a_dict[triple_combination[2]] = triple_combination[3]
		en_triples.append("{}@@@{}@@@{}".format(triple_combination[0], triple_combination[1], triple_combination[2]))
		zh_triples.append("{}@@@{}@@@{}".format(triple_combination[3], triple_combination[4], triple_combination[5]))
	for line in reader_2.readlines():
		line = line.strip()
		triple_combination = re.findall(r"(?<=/)[^/]+(?=>)", line)
		a_dict[triple_combination[3]] = triple_combination[2]
		en_triples.append("{}@@@{}@@@{}".format(triple_combination[3], triple_combination[4], triple_combination[5]))
		zh_triples.append("{}@@@{}@@@{}".format(triple_combination[0], triple_combination[1], triple_combination[2]))

with open("en_zh_alignment.txt", "w", encoding="utf-8") as en_zh_ali_writer, open("zh_en_alignment.txt", "w",
                                                                                  encoding="utf-8") as zh_en_ali_writer, open(
		"en_triples.txt", "w", encoding="utf-8") as et_writer, open("zh_triples.txt", "w",
                                                                    encoding="utf-8") as zt_writer:
	for key in a_dict.keys():
		en_zh_ali_writer.write("{}@@@{}\n".format(key, a_dict[key]))
		zh_en_ali_writer.write("{}@@@{}\n".format(a_dict[key],key))

	for triple in en_triples:
		et_writer.write(triple + "\n")
	for triple in zh_triples:
		zt_writer.write(triple + "\n")