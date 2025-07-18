def count_word_frequency(text):
    words = text.lower().split()
    freq_dict = {}

    for word in words:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1

    return freq_dict

# Example usage
text = "apple banana apple orange banana apple"
result = count_word_frequency(text)
print(result)
