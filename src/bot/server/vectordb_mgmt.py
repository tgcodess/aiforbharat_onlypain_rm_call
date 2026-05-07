'''
Installation :-
pip install indic-num2words chromadb
'''

import chromadb
from num_to_words import num_to_word as num2words
from sentence_transformers import SentenceTransformer

sentenceTransformerModel = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

COLLECTION_NAME = "hindi_numbers"

### Constants
general_rules = [
    ("चौं", "चौ"),
    ("तैं", "तै"),
    ("पैं", "पै"),
    ("त्त", "त"),
    ("प्प", "प"),
    ("क्ख", "ख"),
    ("ग्ग", "ग"),
    ("हज़ार", "हजार"),
    ("सत्तावन", "सतावन"),
    ("चौरासी", "चौरसी"),
    ("उन्नीस", "उनीस"),
    ("इक", "एक"),
    ("उन", "उन्न"),
    ("त्तीस", "तीस"),
    ("सत्तर", "सतर"),
    ("साठ", "सट"),
]

AGE_TEMPLATES = [
    "{w}",
    "{w} साल",
    "मैं {w} साल का हूँ",
    "मेरी उम्र {w} है",
]

COVERAGE_TEMPLATES = [
    "{w}",
    "₹ {w}",
    "{w} का कवरेज",
    "{w} रुपए का कवरेज",
    "{w} कवर",
    "{w} का कवर",
    "{w} रुपए का कवर",
]

###

def initiateVectorDB():
    return chromadb.PersistentClient(path="./vector_db")
    #return chromadb.Client()

def createCollection(client, collectionName):
    return client.get_or_create_collection(name=collectionName)

def insertData(collection, phrases, ids, metadata):
    collection.add(
        documents=phrases,
        embeddings=sentenceTransformerModel.encode(phrases),
        ids=ids,
        metadatas=metadata
    )

def generate_misheard_variants(number, receivedWord, type):
    variants = [receivedWord]
    print(f"Checking {receivedWord}")
    ids = set([f"{number}_{receivedWord}_{receivedWord}"])   # As the word itself is its variant, we add word itself
    numberList = [{"value": number}]

    templates = None
    if type == "age":
        templates = AGE_TEMPLATES
    else:
        templates = COVERAGE_TEMPLATES

    for template in templates:
        word = template.replace("{w}", receivedWord)
        print(f"Preparing {word}")
        
        id = f"{number}_{word}_{word}"
        if id not in ids:
            print(f"id picked : {id}")
            ids.add(id)
            variants.append(word)
            numberList.append({"value": number})
            isInserted = True
        else:
            print(f"{id} not picked")

        for old, new in general_rules:
            if old in word:
                variant = word.replace(old, new)
                id = f"{number}_{word}_{variant}"
                if id not in ids:
                    print(f"id picked : {id}")
                    ids.add(id)
                    variants.append(variant)
                    numberList.append({"value": number})
                    isInserted = True
                else:
                    print(f"{id} not picked")
            if new in word:
                variant = word.replace(new, old)
                id = f"{number}_{word}_{variant}"
                if id not in ids:
                    print(f"id picked : {id}")
                    ids.add(id)
                    variants.append(variant)
                    numberList.append({"value": number})
                    isInserted = True
                else:
                    print(f"{id} not picked")

    return list(variants), list(ids), list(numberList)


def prepareAndInsertDataPerInterval(collection, startRange, endRange, interval, type):

    for i in range(startRange, endRange, interval):
        actualWord = num2words(i, lang='hi', separator="" , combiner="")
        variantList, idList, numberList = generate_misheard_variants(i, actualWord, type) # "सत्तावन साल"

        insertData(collection, variantList, idList, numberList)


def prepareChromaDB():
    # Create Chroma DB
    chromaDBClient = initiateVectorDB()

    # Create collection
    numCollection = createCollection(chromaDBClient, COLLECTION_NAME)

    # Insert numbers, words and associated variations
    print("Going to prepare vector db for 1 to 99")
    prepareAndInsertDataPerInterval(numCollection, 1, 99, 1, "age")
    #print("Going to prepare vector db for 1000 to 1Cr, for every thousand")
    prepareAndInsertDataPerInterval(numCollection, 500000, 600000, 1000, "coverage")
    print("Vector DB prepared")

    return numCollection


def searchWordValue(collection, wordToSearch):

    searchEmbedding = sentenceTransformerModel.encode(wordToSearch)

    #collection = prepareChromaDB()
    result = collection.query(query_embeddings=[searchEmbedding], n_results=3)

    print(result, result["metadatas"][0][0]["value"])

    if result != None and result["metadatas"] and result["metadatas"][0]:
        return result["metadatas"][0][0]["value"]

    return None


#searchWordValue(prepareChromaDB(), "मैं चौतीस का हूँ")