# This is an app for implementing the NLP project on Semantic Search#
import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import torch


def semantic_iterative_search(code_file,query_phrase):
  code_data=[]
  model_name = 'microsoft/codebert-base'
  model = SentenceTransformer(model_name)
  line_numbers=[]
  #print(type(code_file))
  input_file =code_file
  code_file_path = code_file
  with open(code_file_path, 'r') as code_file:
    code_lines = code_file.readlines()

  # Encode the code lines into embeddings
  code_line_embeddings = model.encode(code_lines, convert_to_tensor=True)
  # Perform semantic search
  query = query_phrase
  query_embedding = model.encode([query], convert_to_tensor=True)
  cos_scores = util.pytorch_cos_sim(query_embedding, code_line_embeddings)[0]

  # Sort the results in descending order
  top_results = torch.topk(cos_scores, k=10)

  # Print the top matching code lines
  print("Top 5 matching code lines:")
  for score, idx in zip(top_results[0], top_results[1]):
    temp=[]
    line_numbers.append(idx)
    temp_score =np.round(score.item(),4)
    print(f'Score: {score:.4f}\t Line: {code_lines[idx]}')
    temp.append(temp_score)
    temp.append(idx.item())
    temp_string = code_lines[idx]
    temp.append(temp_string)

    code_data.append(temp)
    print(code_data)
  code_table=pd.DataFrame(code_data, columns=['Relevance Score', 'Line Number ', 'Line'])
  st.write(code_file)
  st.write(code_table)
  semantic_lines=[]
  for element in line_numbers:
     semantic_lines.append(element.item())


  file_name = input_file.split(".")[0].split("/")[-1]
  output_dir = 'C:/Users/Bhavana C/Downloads/Highlighted_Code'
  output_file = output_dir + '/'+ file_name + '_highlighted.html'
  print(output_file)
  lines_to_highlight = semantic_lines
  highlight_lines(input_file, output_file, lines_to_highlight)

def highlight_lines(input_file, output_file, line_numbers):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    highlighted_lines = []
    for line_number, line in enumerate(lines, start=1):
        if line_number in line_numbers:
            line = '<span style="background-color: yellow;">' + line.rstrip() + '</span>'
        highlighted_lines.append(line)

    with open(output_file, 'w') as file:
        file.write('<html><body><pre>')
        file.write(''.join(highlighted_lines))
        file.write('</pre></body></html>')

def find_similarity_scores(query):
    import os
    import torch
    from sentence_transformers import SentenceTransformer, util

    # Load CodeBERT model
    model_name = 'microsoft/codebert-base'
    model = SentenceTransformer(model_name)

    # Path to the directory containing code files
    code_dir = 'C:/Users/Bhavana C/Downloads/Python_files1/'
    # Load code snippets from files
    code_snippets = []
    for file_name in os.listdir(code_dir):
        file_path = os.path.join(code_dir, file_name)
        with open(file_path, 'r') as file:
            code_snippets.append(file.read())

    # Encode code snippets
    encoded_snippets = model.encode(code_snippets)

    # Perform semantic search
    query = "function to construct geomentry"
    query_embedding = model.encode([query])[0]
    results = util.semantic_search(query_embedding, encoded_snippets)

    sim_data=[]
    # Display search results
    for result in results[0]:
        elements=[]
        file_name = os.listdir(code_dir)[result['corpus_id']]
        similarity = result['score']
        elements.append(file_name)
        elements.append(similarity)
        sim_data.append(elements)
        print(f"File: {file_name}\tSimilarity: {similarity}")

    similarity_table = pd.DataFrame(sim_data,columns=['File_Name','Similarity Score'])
    st.write(similarity_table)
def pdf_search(query):
    import spacy
    import string
    import gensim
    import operator
    import re
    import locale
    locale.getpreferredencoding = lambda: "UTF-8"

    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from io import StringIO

    queries=[]
    queries.append(query)
    def convert_pdf_to_txt(path):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()
        return text

    import os
    directory = 'C:/Users/Bhavana C/Downloads/NLP_pdfs'

    text = []

    for filename in os.listdir(directory):
        # f = os.path.join(directory, filename)
        path = directory + '/' + filename
        print(path)
        t = convert_pdf_to_txt(path)
        text.append(t)

    from spacy.lang.en.stop_words import STOP_WORDS

    spacy_nlp = spacy.load('en_core_web_sm')

    # create list of punctuations and stopwords
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    # function for data cleaning and processing
    # This can be further enhanced by adding / removing reg-exps as desired.

    def spacy_tokenizer(sentence):

        # remove distracting single quotes
        sentence = re.sub('\'', '', sentence)

        # remove digits adnd words containing digits
        sentence = re.sub('\w*\d\w*', '', sentence)

        # replace extra spaces with single space
        sentence = re.sub(' +', ' ', sentence)

        # remove unwanted lines starting from special charcters
        sentence = re.sub(r'\n: \'\'.*', '', sentence)
        sentence = re.sub(r'\n!.*', '', sentence)
        sentence = re.sub(r'^:\'\'.*', '', sentence)

        # remove non-breaking new line characters
        sentence = re.sub(r'\n', ' ', sentence)

        # remove punctunations
        sentence = re.sub(r'[^\w\s]', ' ', sentence)

        # creating token object
        tokens = spacy_nlp(sentence)

        # lower, strip and lemmatize
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]

        # remove stopwords, and exclude words less than 2 characters
        tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]

        # return tokens
        return tokens

    tokenized = []
    print('Cleaning and Tokenizing...')
    for ele in text:
        token = spacy_tokenizer(ele)
        tokenized.append(token)

    from gensim import corpora

    # creating term dictionary

    dictionary = corpora.Dictionary(tokenized)

    stoplist = set('hello and if this can would should could tell ask stop come go')
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    dictionary.filter_tokens(stop_ids)

    corpus = [dictionary.doc2bow(desc) for desc in tokenized]

    word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]

    ambertag_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
    ambertag_lsi_model = gensim.models.LsiModel(ambertag_tfidf_model[corpus], id2word=dictionary, num_topics=300)

    gensim.corpora.MmCorpus.serialize('ambertag_tfidf_model_mm', ambertag_tfidf_model[corpus])
    gensim.corpora.MmCorpus.serialize('ambertag_lsi_model_mm', ambertag_lsi_model[ambertag_tfidf_model[corpus]])

    # Load the indexed corpus
    ambertag_tfidf_corpus = gensim.corpora.MmCorpus('ambertag_tfidf_model_mm')
    ambertag_lsi_corpus = gensim.corpora.MmCorpus('ambertag_lsi_model_mm')

    from gensim.similarities import MatrixSimilarity
    ambertag_index = MatrixSimilarity(ambertag_lsi_corpus, num_features=ambertag_lsi_corpus.num_terms)

    index = []

    title = ''

    from operator import itemgetter

    def search_similar_words(search_term):
        global index

        query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
        query_tfidf = ambertag_tfidf_model[query_bow]
        query_lsi = ambertag_lsi_model[query_tfidf]
        ambertag_index.num_best = 10

        ambertag_list = ambertag_index[query_lsi]
        ambertag_list.sort(key=itemgetter(1), reverse=True)
        ambertag_names = []
        index = ambertag_list
        print(index)

        for j, ambertag in enumerate(ambertag_list):

            ambertag_names.append(
                {
                    'Relevance': round((ambertag[1] * 100), 2),
                    # including just the title of the document in the column
                    'Relevant documents': text[ambertag[0]].split('\n')[0]
                }

            )

            if j == (ambertag_index.num_best - 1):
                break

        return pd.DataFrame(ambertag_names, columns=['Relevance', 'Relevant documents'])

    df = search_similar_words(query)
    st.write(df)

    from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    documents = []

    import os
    directory = 'C:/Users/Bhavana C/Downloads/NLP_pdfs'
    # directory = 'gdrive/My Drive/gs'
    text = []

    for filename in os.listdir(directory):
        # f = os.path.join(directory, filename)
        path = directory + '/' + filename
        print(path)
        documents.append(path)
        t = convert_pdf_to_txt(path)
        text.append(t)

    # data pre processing
    import re

    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    doc_text = []
    for i in range(len(text)):
        texting = decontracted(text[i])
        texting = texting.replace('\\r', ' ')
        texting = texting.replace('\\"', ' ')
        texting = texting.replace('\\n', ' ')
        doc_text.append(texting)

    for i in range(len(doc_text)):
        doc_text[i] = doc_text[i].split()

    sentence_embeddings = []
    for i in range(len(doc_text)):
        sentence_embeddings.append(model.encode(doc_text[i], convert_to_tensor=True))

    from sentence_transformers import SentenceTransformer, util
    import torch
    import fitz

    semantic_words = []

    list_word_pages = []
    strings = []
    # final_data={}
    top_k = 30
    top_results = []
    for query in queries:

        semantic_words = []
        query_embedding = model.encode(query, convert_to_tensor=True)
        i = 0
        for ele in documents:
            # if ele == 'Company_Assets_Usage_policy.pdf':
            # continue
            list1 = []
            list1.append(query)
            list1.append(ele)
            # Semantic_df['Document_title'] = ele
            word_pages = {}
            page_nos = []
            doc = fitz.open(ele)
            cos_scores = util.cos_sim(query_embedding, sentence_embeddings[i])[0]
            top_results = torch.topk(cos_scores, k=top_k)

            for score, idx in zip(top_results[0], top_results[1]):
                # print(i)
                # print(doc_text[i])
                # print(len(doc_text[i]))
                # print(idx)
                # print(doc_text[i][idx], "(Score: {:.4f})".format(score))
                if doc_text[i][idx] not in semantic_words:
                    semantic_words.append(doc_text[i][idx])
            list1.append(semantic_words)

            for word in semantic_words:
                for page in doc:
                    texter = word
                    text_instances = page.search_for(texter)
                    if text_instances:
                        page_nos.append(page)
                        word_pages[word] = page
                        # word_pages.update({word:page})
                        # print(word_pages)

                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()

            list1.append(word_pages)
            output_data = str(list1).encode('ascii', 'ignore').decode('ascii')

            with open("C:/Users/Bhavana C/Downloads/Data2.txt", "a") as f:
                for item in list1:
                    x = str(item).encode('ascii', 'ignore').decode('ascii')
                    strings.append(x)
                    print(x)
                    f.write(x)
                    # f.write('\n')
                f.write('\n')
            list_word_pages.append(list1)
            # print(list_word_pages)
            # print(list1)
            # print(page_nos)

            # list_word_pages.append(word_pages)
            i = i + 1
            main_dir = "C:/Users/Bhavana C/Downloads/Output_NLP_pdfs1/"
            output_path = ele.split("/")[-1].split(".")[-2] + "-output.pdf"
            output_file = main_dir + output_path
            doc.save(output_file, garbage=4, deflate=True, clean=True)

    Semantic_df = pd.DataFrame(columns=['Query_word', 'Document_title', 'Semantic_words', 'Page_Numbers'])

    count = 0
    j = 0
    while j <= 38:
        # print(j)
        Semantic_df.loc[count, 'Query_word'] = strings[j]
        Semantic_df.loc[count, 'Document_title'] = strings[j + 1]
        Semantic_df.loc[count, 'Semantic_words'] = strings[j + 2]
        Semantic_df.loc[count, 'Page_Numbers'] = strings[j + 3]
        j = j + 4
        count = count + 1
        if j == 37:
            break

    st.write(Semantic_df)

def find_github(query_phrase):
    import requests
    import base64

    model_name = 'microsoft/codebert-base'
    model = SentenceTransformer(model_name)
    line_numbers = []
    repo_owner = "Bhavana-datascientist"
    repo_name = "NLP_test"
    base_url = "https://api.github.com"
    access_token = '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'
    base_url1 = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents'
    headers = {'Authorization': f'token {access_token}'}

    response = requests.get(base_url1, headers=headers)
    files = response.json()
    for file_info in files:
        # print(file_info['name'])
        code_data = []
        file_path = file_info['name']
        if '.py' in file_path:
            api_url = f"{base_url}/repos/{repo_owner}/{repo_name}/contents/{file_path}"
            response = requests.get(api_url)
            data = response.json()
            git_text = base64.b64decode(data['content']).decode('utf8').split("\n")
            code_lines = git_text
            # Encode the code lines into embeddings
            code_line_embeddings = model.encode(code_lines, convert_to_tensor=True)
            # Perform semantic search
            query = query_phrase
            query_embedding = model.encode([query], convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, code_line_embeddings)[0]

            # Sort the results in descending order
            top_results = torch.topk(cos_scores, k=10)

            # Print the top matching code lines
            print("Top 5 matching code lines:")
            for score, idx in zip(top_results[0], top_results[1]):
                temp = []
                line_numbers.append(idx)
                temp_score = np.round(score.item(), 4)
                print(f'Score: {score:.4f}\t Line: {code_lines[idx]}')
                temp.append(temp_score)
                temp.append(idx.item())
                temp_string = code_lines[idx]
                temp.append(temp_string)

                code_data.append(temp)
                print(code_data)
            code_table = pd.DataFrame(code_data, columns=['Relevance Score', 'Line Number ', 'Line'])
            st.write(file_path)
            st.write(code_table)
            semantic_lines = []
            for element in line_numbers:
                semantic_lines.append(element.item())

            file_name = file_path
            output_dir = 'D:/NLP_output/Highlighted_Code'
            output_file = output_dir + '/' + file_name + '_highlighted.html'
            print(output_file)
            lines_to_highlight = semantic_lines
            #downloading file from github to highlight
            base_url_git = "https://api.github.com/repos/{}/{}/contents/{}".format(repo_owner, repo_name, file_path)
            headers_git = {
                "Authorization": f"token {access_token}"
            }
            response_git = requests.get(base_url, headers=headers_git)
            if response_git.status_code == 200:
                json_response = response.json()
                download_url = json_response["download_url"]
            else:
                print("Failed to fetch the file. Status code:", response_git.status_code)
                exit()
            file_response = requests.get(download_url)
            if file_response.status_code == 200:
                # Replace 'file.py' with your desired local file name.

                local_file_name = "D:/NLP_output/Git_Downloaded_files/"+file_path
                with open(local_file_name, "wb+") as file:
                    file.write(file_response.content)
                print(f"File downloaded and saved as {local_file_name}.")
            else:
                print("Failed to download the file. Status code:", file_response.status_code)

            input_gitfile = "D:/NLP_output/Git_Downloaded_files"+'/'+file_path
            f_name = file_path.split(".")[0].split("/")[-1]
            output_gitfile = "D:/NLP_output/Git_Highlighted_code"+"/"+f_name+'_highlighted.html'
            highlight_lines(input_gitfile, output_gitfile, lines_to_highlight)
            #################
            #highlighting lines
            highlighted_lines = []
            lines = git_text
            for line_number, line in enumerate(lines, start=1):
                if line_number in line_numbers:
                    line = '<span style="background-color: yellow;">' + line.rstrip() + '</span>'
                highlighted_lines.append(line)

            with open(output_file, 'w') as file:
                file.write('<html><body><pre>')
                file.write(''.join(highlighted_lines))
                file.write('</pre></body></html>')


def main():
    st.title(":red[Semantic Search]")
    html_temp = """
    <div style ="background-color:tomato; padding:10px"
    <h2 style = "color:white;text-align:center;">Performs Semantic Search on Documents and Code files</h2>
    </div>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    query = st.text_input("Query","Type your query word here")
    result =""
    #if st.button("Find Similarity Scores in Documents"):
        #find_similarity_scores(query)
    if st.button("Find Semantic Words and Phrases in Source Code repository"):
        find_similarity_scores(query)
        directory = 'C:/Users/Bhavana C/Downloads/Python_files1/'
        query_phrase = query
        #query_phrase = 'function to construct geometry'
        for filename in os.listdir(directory):
            path = directory + '/' + filename
            print(type(path))
            semantic_iterative_search(path, query_phrase)
    if st.button('Find semantic words and phrases in PDF documents'):
        pdf_search(query)

    if st.button('Find Semantic words and phrases on Github Source coderepository'):
        find_github(query)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
