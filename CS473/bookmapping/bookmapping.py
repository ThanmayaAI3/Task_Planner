import os
import nltk
import argparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models, similarities
import csv

def process_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]

    return tokens

def main(input_file):
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    with open(input_file, 'r') as f:
        path_book1 = f.readline()
        path_book2 = f.readline()
        path_book1 = ''.join(path_book1.split('\n'))
        path_book2 = ''.join(path_book2.split('\n'))
        #print(path_book1)
        #print(path_book2)

    # Gathering text files
    book1_chapters = []
    book1_chapter_names = []
    for filename in os.listdir(path_book1):
        if filename.endswith('.txt'):
            with open(os.path.join(path_book1, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = process_text(text)
                book1_chapters.append(tokens)
                book1_chapter_names.append(filename)

    book2_chapters = []
    book2_chapter_names = []
    for filename in os.listdir(path_book2):
        if filename.endswith('.txt'):
            with open(os.path.join(path_book2, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = process_text(text)
                book2_chapters.append(tokens)
                book2_chapter_names.append(filename)

    all_docs = book1_chapters + book2_chapters
    dictionary = corpora.Dictionary(all_docs)
    #print(dictionary)
    corpus = [dictionary.doc2bow(doc) for doc in all_docs]
    tf_idf = models.TfidfModel(corpus, smartirs='ltc')
    tfidf_corpus = [tf_idf[doc] for doc in corpus]

    book1_tfidf = tfidf_corpus[:len(book1_chapters)]
    book2_tfidf = tfidf_corpus[len(book1_chapters):]

    dict_length = len(dictionary)
    index = similarities.MatrixSimilarity(book2_tfidf, num_features=dict_length)
    output = []

    # Similarity calculations
    for i, doc_tfidf in enumerate(book1_tfidf):
        sims = index[doc_tfidf]
        most_similar_idx = sims.argmax()
        similarity_score = sims[most_similar_idx]
        chapter_book1 = book1_chapter_names[i]
        chapter_book2 = book2_chapter_names[most_similar_idx]
        output.append((chapter_book1, chapter_book2, similarity_score))

    # Write output to output.csv
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in output:
            # Round the similarity score to two decimal places
            writer.writerow([row[0], row[1], row[2]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map chapters between two books.")
    parser.add_argument("input_file", help="Path to the input.txt file")
    args = parser.parse_args()
    main(args.input_file)
