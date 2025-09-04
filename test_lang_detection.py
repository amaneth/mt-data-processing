#!/usr/bin/env python3
import fasttext
import os
from langdetect import detect, detect_langs, DetectorFactory
from transformers import pipeline
from pyfranc import franc

# Add Lingua model
from lingua import Language, LanguageDetectorBuilder

# Initialize Lingua detector with supported languages
try:
    lingua_languages = [Language.ENGLISH, Language.AFRIKAANS, Language.SOMALI,
                        Language.SWAHILI, Language.ZULU, Language.HAUSA]
    lingua_detector = LanguageDetectorBuilder.from_languages(*lingua_languages).build()
    lingua = True
    print("✅ Lingua detector loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Lingua detector: {e}")
    lingua_detector = None
    lingua = False
# Function to detect language using AfroLM
def afrolm_language_detect(text, model, tokenizer, target_languages=None):
    """Fallback pattern-based language detection"""
    return pattern_based_language_detection(text)

def pattern_based_language_detection(text, base_confidence=0.5):
    """Fallback pattern-based language detection"""
    scores = {}

    text_lower = text.lower()

    # Language-specific patterns and keywords
    patterns = {
        'en': ['the ', 'and ', 'is ', 'of ', 'in ', 'to ', 'for ', 'on ', 'with ', 'at '],
        'af': ['die ', 'en ', 'van ', 'met ', 'op ', 'vir ', 'wat ', 'jy ', 'nie '],
        'so': ['waa ', 'ka ', 'ku ', 'ee ', 'waay', 'leh', 'laa ', 'aan '],
        'am': ['እኔ', 'አትሆነም', 'ሰላም', 'አልፎ', 'አልፈጸመም'],
        'sw': ['na ', 'kwa ', 'ya ', 'ni ', 'wa ', 'katika ', 'hadi ', 'juu ', 'chini '],
        'zu': ['ng', 'ku ', 'nga ', 'pho ', 'hle '],
        'ha': ['ce ', 'da ', 'ne ', 'ta ', 'kuma ', 'ba ', 'ko ']
    }

    for lang, keywords in patterns.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
        if score > 0:
            scores[lang] = score / len(keywords)

    # Boost confidence with base_confidence
    if scores:
        max_lang = max(scores, key=scores.get)
        max_score = scores[max_lang]
        adjusted_score = min(1.0, max_score * base_confidence + (base_confidence * 0.5))
        return [(max_lang, adjusted_score)]
    else:
        return [('und', 0.0)]

# Add AfroLM model
from transformers import XLMRobertaTokenizer, XLMRobertaModel
try:
    # AfroLM is XLM-Roberta based, load as specified in the repository
    afrolm_tokenizer = XLMRobertaTokenizer.from_pretrained("bonadossou/afrolm_active_learning")
    afrolm_tokenizer.model_max_length = 256
    afrolm_model = XLMRobertaModel.from_pretrained("bonadossou/afrolm_active_learning")
    print("✅ AfroLM model loaded successfully")
    afrolm = True
except Exception as e:
    print(f"❌ Failed to load AfroLM model: {e}")
    afrolm = None
    afrolm_model = None
    afrolm_tokenizer = None

# Set seed for consistent results
DetectorFactory.seed = 0

# Sample sentences to test - Only these languages: af, amh, sw, so, zu, ha, en
english_sentences = [
    "Hello, how are you today?",
    "I am learning machine translation.",
    "The weather is beautiful today.",
    "What is your name?",
    "We are building a new system.",
    "The global economy is undergoing significant transformations due to technological advancements and changing consumer behavior patterns around the world, which require careful analysis by economists and policymakers alike.",
    "I'm really excited to be working on this project with my colleagues from different departments, as it allows us to collaborate on innovative solutions for our customers' complex needs and challenges.",
    "Yesterday I visited the museum where I saw ancient artifacts from various civilizations, including Egyptian pharaohs' treasures, Greek sculptures, and Roman pottery that tell fascinating stories of human history.",
    "The scientific community has made remarkable progress in understanding quantum mechanics, but there are still many unanswered questions about the nature of reality, time, and the universe itself.",
    "Different cultures across the globe have unique traditions for celebrating the harvest season, such as Día de los Muertos in Mexico, Thanksgiving in the United States, and Diwali in India, each reflecting distinctive historical and social contexts."
]

afrikaans_sentences = [
    "Hallo, hoe gaan dit vandag met jou?",
    "Ek leer masjienvertaling.",
    "Die weer is pragtig vandag.",
    "Wat is jou naam?",
    "Ons bou 'n nuwe stelsel.",
    "Dit is baie interessant om te sien hoe die tegnologie die manier verander waarop ons kommunikeer en ons daaglikse lewe beïnvloed, veral in ontwikkelende lande soos Afrika.",
    "Die Afrikaanse taal het 'n ryk geskiedenis en word gebruik in verskillende kontekste, van literatuur tot wetenskap, maar ons moet werk daaraan om dit lewendig te hou vir toekomstige geslagte.",
    "Ek het gehoor dat die ekonomie van Suid-Afrika baie verander het die afgelope paar jaar, hoofsaaklik as gevolg van beleggings in nuwe industrië en verbeterings in infrastruktuur.",
    "Die onderwyser het gesê dat dit belangrik is om na ander kulture te luister en te leer van hul ervarings om 'n meer diverse en inklusiewe gemeenskap te bou.",
    "In die toekoms sal ons meer moet fokus op omgewingsbewaring om die planeet te beskerm teen klimaatsverandering en sy gevolge."
]

somali_sentences = [
    "Maamulka Gobolka Banaadir oo Dugsiyada Muqdisho faray in la suro Calanka Somalia.",
    "Ma van Roodepoort 58 kg ligter: Klomp klein veranderinge vir 'n gróót verskil",
    "Salaan, sidee tahay maanta?",
    "Waan jeclahay in aan wax badan barto ku saabsan teknolojiyada casriga ah, iyada oo ku xiran xawaaraha internetka ee hooseeya, laakiin waxaa jira horumarro waaweyn oo dhaqaale ah oo ku yimaada.",
    "Jamhuuriyadda Soomaaliya waa waddan ku yaala Geeska Afrika oo leh taariikh dheer iyo dhaqamo kala duwan oo ka turjumaya wanaagga dadkeeda.",
    "Anigu waxaan aaminsanahay in horumarinta teknolojiyada ay door muhimka ah ka cayaarto mustaqbalka Afrika, gaar ahaan Soomaaliya, halka ay ku lug leedahay bixinta waxbarashada tayo sare leh.",
    "Dhaqaalaha Afrika wuxuu u muuqdaa mid sare u kaca laakiin wali waxaa jira caqabado maaliyadeed oo badan oo dib u maamirta.",
    "Taalimidda waxaa jiri doona horumarro, laakiin waa inaan adkeyno dhaqankeenna dastuurka ku xafsaday si aan u horumarno."
]

amharic_sentences = [
    "ሰላም፣ እንዴት ነህ ዛሬ?",
    "አበበ በሶ በላይ እንደሚታወቀው የአማርኛ ቋንቋ ነው።",
    "አበበ",
    "የኢትዮጵያ ፕሪምየር ሊግ የ2014 የውድድር ዘመን ዛሬ ተጀምሯል",
    "እኔ አቀበያለሁ እንደሆነልና በኢትዮጵያ ለተለመዱ ቦታያጋሬሮች ጓደኞች እለም በካቢኔት መስብየት በራሱ ነው።",
    "በጣም እንደሚታዉቀ የተለመዱ ለተለመዱ ኢትዮጵያ ለሚያልፈልጉ ለሚኑ ለመስብየት ለመኖር ለመስብየት ምን እንደሚቆይ አልተለምዱም።",
    "የኢትዮጵያ ሚኒስትሮች ብሔራዊ ልምዱ ገዳይ እንደሚኑ ለመኑ ለሚያልፈልጉ ለመስብየት ለመኖር ለመስብየት አልተለምዱም።",
    "እኔ አቀበያለሁ እንደሆነልና ለኢትዮጵያ ለሚያልፈልጉ ለሚኑ ለመስብየት ለመኖር ለመስብየት አልተለምዱም።",
    "የኢትዮጵያ ለሚያልፈልጉ ለሚኑ ለመስብየት ለመኖር ለመስብየት አልተለምዱም።"
]

swahili_sentences = [
    "Habari, habari ya leo?",
    "Ninajifunza tafsiri ya mashine.",
    "Hali ya hewa ni nzuri leo.",
    "Jina lako nani?",
    "Ninataka kujifunza lugha mpya ili kuweza kuwasiliana na watu kutoka sehemu mbalimbali duniani kuhusu usafiri, teknolojia na elimu.",
    "Nchi za Afrika Mashariki zimefanya maendeleo makubwa katika sekta ya teknolojia na biashara, iliyoambatana na ukuaji wa uchumi.",
    "Wanasayansi wa riadha walifanya utafiti kuhusu athari za mablentu kwa mtu na wabunifu watu wa kulelea ya afya ya binadamu.",
    "Mkwalo wa kuandika lugha wakati wa zamani ulikuwa na vifupisho na mikanganya ya muundo wa nuksi, iliyotaka maelezo zaidi.",
    "Katika ulimwengu wa siku hizi, uboreshaji wa mifumo ya elimu ni muhimu ili kuhakikisha wanafunzi wanapata elimu bora na fursa ya maisha."
]

zulu_sentences = [
    "Sawubona, unjani namhlanje?",
    "Ngifunda ukuhumusha kwemishini.",
    "Isimo sezulu sihle namhlanje.",
    "Ubani igama lakho?",
    "Sakha isistimu entsha.",
    "Ngithanda ukufunda izilimi ezintsha ukuze ngikwazi ukuxhumana nabantu abavela kwezinye izindawo emhlabeni.",
    "Abantu ba-Afrika Baphumelela ukufunda amaPc ukuze baguqule imicabango yawo kuyinto enobuholi ekhomputha.",
    "Isibano sokuqala lesifunda esibhabhatusi sikhoma ngekhambi ngezikhathi ezinhluqo, esiphinde sibuyele kuguquko oluqhubekayo.",
    "Amakhono okuqonda amaBantu ase-Afrika kubalulekile ukuthi ama-nostalgia afike ngaphezulu kokulungisa.",
    "Izinkezele ezintsha zokuqala zingakhefa ukufunda ama-lamente akudala, kodwa zingavumela ukufinyelela emfuleni."
]

hausa_sentences = [
    "Sannu, yaya kake yau?",
    "Ina koyi fassarar mashina.",
    "Yanayin zuwa yana da kyau a yau.",
    "Menene sunanka?",
    "Muna gina sabuwar tsarin.",
    "Ina zamna don koyi inda a fis da masu huguma da gonar da ya na ga ni.",
    "Masana da masu koyi tafsiri da yau suna koyi a lakasida don koyi fassarar kudi da lokaci da kuda ga ni.",
    "Masu kibbi da masu hali da yau suna bana ga koyi lokaci da kudi da yau da ya na ga ni.",
    "Masu koyi a wano suna koyi lokaci da kudi da yau da ya na ga ni da masu huguma da gonar da ya na ga ni.",
    "Masu koyi a wano suna koyi lokaci da kudi da yau da ya na ga ni da masu huguma da gonar da ya na ga ni."
]

print("Testing Langdetect Language Detection...")
print("=" * 60)

print("\nTesting English sentences with langdetect:")
print("-" * 40)
for i, sentence in enumerate(english_sentences, 1):
    try:
        lang = detect(sentence)
        langs = detect_langs(sentence)
        print(f"EN {i}: '{sentence[:50]}...'")
        print(f"      Detected: {lang} | Probabilities: {langs}")
    except Exception as e:
        print(f"EN {i}: '{sentence[:50]}...' - Error: {e}")

print("\nTesting Somali sentences with langdetect:")
print("-" * 40)
for i, sentence in enumerate(somali_sentences, 1):
    try:
        lang = detect(sentence)
        langs = detect_langs(sentence)
        print(f"SO {i}: '{sentence[:50]}...'")
        print(f"      Detected: {lang} | Probabilities: {langs}")
    except Exception as e:
        print(f"SO {i}: '{sentence[:50]}...' - Error: {e}")

print("\nTesting Amharic sentences with langdetect:")
print("-" * 40)
for i, sentence in enumerate(amharic_sentences, 1):
    try:
        lang = detect(sentence)
        langs = detect_langs(sentence)
        print(f"AM {i}: '{sentence[:50]}...'")
        print(f"      Detected: {lang} | Probabilities: {langs}")
    except Exception as e:
        print(f"AM {i}: '{sentence[:50]}...' - Error: {e}")

print("\nTesting Afrikaans sentences with langdetect:")
print("-" * 40)
for i, sentence in enumerate(afrikaans_sentences, 1):
    try:
        lang = detect(sentence)
        langs = detect_langs(sentence)
        print(f"AF {i}: '{sentence[:50]}...'")
        print(f"      Detected: {lang} | Probabilities: {langs}")
    except Exception as e:
        print(f"AF {i}: '{sentence[:50]}...' - Error: {e}")

print("\nTesting Swahili sentences with langdetect:")
print("-" * 40)
for i, sentence in enumerate(swahili_sentences, 1):
    try:
        lang = detect(sentence)
        langs = detect_langs(sentence)
        print(f"SW {i}: '{sentence[:50]}...'")
        print(f"      Detected: {lang} | Probabilities: {langs}")
    except Exception as e:
        print(f"SW {i}: '{sentence[:50]}...' - Error: {e}")

print("\nTesting Zulu sentences with langdetect:")
print("-" * 40)
for i, sentence in enumerate(zulu_sentences, 1):
    try:
        lang = detect(sentence)
        langs = detect_langs(sentence)
        print(f"ZU {i}: '{sentence[:50]}...'")
        print(f"      Detected: {lang} | Probabilities: {langs}")
    except Exception as e:
        print(f"ZU {i}: '{sentence[:50]}...' - Error: {e}")

print("\nTesting Hausa sentences with langdetect:")
print("-" * 40)
for i, sentence in enumerate(hausa_sentences, 1):
    try:
        lang = detect(sentence)
        langs = detect_langs(sentence)
        print(f"HA {i}: '{sentence[:50]}...'")
        print(f"      Detected: {lang} | Probabilities: {langs}")
    except Exception as e:
        print(f"HA {i}: '{sentence[:50]}...' - Error: {e}")

print("\n" + "=" * 60)


print("Testing FastText Language Detection for Somali Dataset...")
print("=" * 60)

# Load FastText model
try:
    model = fasttext.load_model('lid.176.bin')
    print("✅ FastText model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load FastText model: {e}")
    exit(1)

print("\nTesting English sentences:")
print("-" * 40)
for i, sentence in enumerate(english_sentences, 1):
    predictions = model.predict(sentence, k=3)  # Get top 3 predictions
    lang_codes = [pred.replace("__label__", "") for pred in predictions[0]]
    scores = [f"{score:.3f}" for score in predictions[1]]
    print(f"EN {i}: '{sentence[:50]}...'")
    print(f"      Detected: {lang_codes[0]} ({scores[0]}) | Alternatives: {', '.join([f'{c}({s})' for c,s in zip(lang_codes[1:], scores[1:])])}")

print("\nTesting Somali sentences:")
print("-" * 40)
for i, sentence in enumerate(somali_sentences, 1):
    predictions = model.predict(sentence, k=3)  # Get top 3 predictions
    lang_codes = [pred.replace("__label__", "") for pred in predictions[0]]
    scores = [f"{score:.3f}" for score in predictions[1]]
    print("SO {}: '{}'...".format(i, sentence[:50]))
    print("      Detected: {} ({}) | Alternatives: {}".format(lang_codes[0], scores[0], ', '.join(["{}({})".format(c,s) for c,s in zip(lang_codes[1:], scores[1:])])))

print("\nTesting Amharic sentences:")
print("-" * 40)
for i, sentence in enumerate(amharic_sentences, 1):
    predictions = model.predict(sentence, k=3)
    lang_codes = [pred.replace("__label__", "") for pred in predictions[0]]
    scores = [f"{score:.3f}" for score in predictions[1]]
    print("AM {}: '{}'...".format(i, sentence[:50]))
    print("      Detected: {} ({}) | Alternatives: {}".format(lang_codes[0], scores[0], ', '.join(["{}({})".format(c,s) for c,s in zip(lang_codes[1:], scores[1:])])))

print("\nTesting Afrikaans sentences:")
print("-" * 40)
for i, sentence in enumerate(afrikaans_sentences, 1):
    predictions = model.predict(sentence, k=3)
    lang_codes = [pred.replace("__label__", "") for pred in predictions[0]]
    scores = [f"{score:.3f}" for score in predictions[1]]
    print("AF {}: '{}'...".format(i, sentence[:50]))
    print("      Detected: {} ({}) | Alternatives: {}".format(lang_codes[0], scores[0], ', '.join(["{}({})".format(c,s) for c,s in zip(lang_codes[1:], scores[1:])])))

print("\nTesting Swahili sentences:")
print("-" * 40)
for i, sentence in enumerate(swahili_sentences, 1):
    predictions = model.predict(sentence, k=3)
    lang_codes = [pred.replace("__label__", "") for pred in predictions[0]]
    scores = [f"{score:.3f}" for score in predictions[1]]
    print("SW {}: '{}'...".format(i, sentence[:50]))
    print("      Detected: {} ({}) | Alternatives: {}".format(lang_codes[0], scores[0], ', '.join(["{}({})".format(c,s) for c,s in zip(lang_codes[1:], scores[1:])])))

print("\nTesting Zulu sentences:")
print("-" * 40)
for i, sentence in enumerate(zulu_sentences, 1):
    predictions = model.predict(sentence, k=3)
    lang_codes = [pred.replace("__label__", "") for pred in predictions[0]]
    scores = [f"{score:.3f}" for score in predictions[1]]
    print("ZU {}: '{}'...".format(i, sentence[:50]))
    print("      Detected: {} ({}) | Alternatives: {}".format(lang_codes[0], scores[0], ', '.join(["{}({})".format(c,s) for c,s in zip(lang_codes[1:], scores[1:])])))

print("\nTesting Hausa sentences:")
print("-" * 40)
for i, sentence in enumerate(hausa_sentences, 1):
    predictions = model.predict(sentence, k=3)
    lang_codes = [pred.replace("__label__", "") for pred in predictions[0]]
    scores = [f"{score:.3f}" for score in predictions[1]]
    print("HA {}: '{}'...".format(i, sentence[:50]))
    print("      Detected: {} ({}) | Alternatives: {}".format(lang_codes[0], scores[0], ', '.join(["{}({})".format(c,s) for c,s in zip(lang_codes[1:], scores[1:])])))

print("\n" + "=" * 60)


print("Testing AfroLID Language Detection...")
print("=" * 60)

# Load AfroLID model
try:
    afrolid = pipeline("text-classification", model='UBC-NLP/afrolid_1.5')
    print("✅ AfroLID model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load AfroLID model: {e}")
    exit(1)

print("\nTesting English sentences with AfroLID:")
print("-" * 40)
for i, sentence in enumerate(english_sentences, 1):
    try:
        result = afrolid(sentence)
        lang = result[0]['label']
        score = result[0]['score']
        print("EN {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.2f}%".format(lang, score * 100))
    except Exception as e:
        print("EN {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Somali sentences with AfroLID:")
print("-" * 40)
for i, sentence in enumerate(somali_sentences, 1):
    try:
        result = afrolid(sentence)
        lang = result[0]['label']
        score = result[0]['score']
        print("SO {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.2f}%".format(lang, score * 100))
    except Exception as e:
        print("SO {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Amharic sentences with AfroLID:")
print("-" * 40)
for i, sentence in enumerate(amharic_sentences, 1):
    try:
        result = afrolid(sentence)
        lang = result[0]['label']
        score = result[0]['score']
        print("AM {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.2f}%".format(lang, score * 100))
    except Exception as e:
        print("AM {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Afrikaans sentences with AfroLID:")
print("-" * 40)
for i, sentence in enumerate(afrikaans_sentences, 1):
    try:
        result = afrolid(sentence)
        lang = result[0]['label']
        score = result[0]['score']
        print("AF {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.2f}%".format(lang, score * 100))
    except Exception as e:
        print("AF {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Swahili sentences with AfroLID:")
print("-" * 40)
for i, sentence in enumerate(swahili_sentences, 1):
    try:
        result = afrolid(sentence)
        lang = result[0]['label']
        score = result[0]['score']
        print("SW {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.2f}%".format(lang, score * 100))
    except Exception as e:
        print("SW {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Zulu sentences with AfroLID:")
print("-" * 40)
for i, sentence in enumerate(zulu_sentences, 1):
    try:
        result = afrolid(sentence)
        lang = result[0]['label']
        score = result[0]['score']
        print("ZU {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.2f}%".format(lang, score * 100))
    except Exception as e:
        print("ZU {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Hausa sentences with AfroLID:")
print("-" * 40)
for i, sentence in enumerate(hausa_sentences, 1):
    try:
        result = afrolid(sentence)
        lang = result[0]['label']
        score = result[0]['score']
        print("HA {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.2f}%".format(lang, score * 100))
    except Exception as e:
        print("HA {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\n" + "=" * 60)


print("Testing PyFranc Language Detection...")
print("=" * 60)

print("\nTesting English sentences with PyFranc:")
print("-" * 40)
for i, sentence in enumerate(english_sentences, 1):
    try:
        result = franc.lang_detect(sentence)
        lang = result[0][0]
        score = result[0][1]
        print("EN {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.3f}".format(lang, score))
    except Exception as e:
        print("EN {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Somali sentences with PyFranc:")
print("-" * 40)
for i, sentence in enumerate(somali_sentences, 1):
    try:
        result = franc.lang_detect(sentence)
        lang = result[0][0]
        score = result[0][1]
        print("SO {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.3f}".format(lang, score))
    except Exception as e:
        print("SO {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Amharic sentences with PyFranc:")
print("-" * 40)
for i, sentence in enumerate(amharic_sentences, 1):
    try:
        result = franc.lang_detect(sentence)
        lang = result[0][0]
        score = result[0][1]
        print("AM {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.3f}".format(lang, score))
    except Exception as e:
        print("AM {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Afrikaans sentences with PyFranc:")
print("-" * 40)
for i, sentence in enumerate(afrikaans_sentences, 1):
    try:
        result = franc.lang_detect(sentence)
        lang = result[0][0]
        score = result[0][1]
        print("AF {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.3f}".format(lang, score))
    except Exception as e:
        print("AF {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Swahili sentences with PyFranc:")
print("-" * 40)
for i, sentence in enumerate(swahili_sentences, 1):
    try:
        result = franc.lang_detect(sentence)
        lang = result[0][0]
        score = result[0][1]
        print("SW {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.3f}".format(lang, score))
    except Exception as e:
        print("SW {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Zulu sentences with PyFranc:")
print("-" * 40)
for i, sentence in enumerate(zulu_sentences, 1):
    try:
        result = franc.lang_detect(sentence)
        lang = result[0][0]
        score = result[0][1]
        print("ZU {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.3f}".format(lang, score))
    except Exception as e:
        print("ZU {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\nTesting Hausa sentences with PyFranc:")
print("-" * 40)
for i, sentence in enumerate(hausa_sentences, 1):
    try:
        result = franc.lang_detect(sentence)
        lang = result[0][0]
        score = result[0][1]
        print("HA {}: '{}'...".format(i, sentence[:50]))
        print("      Detected: {} | Score: {:.3f}".format(lang, score))
    except Exception as e:
        print("HA {}: '{}'...".format(i, sentence[:50]))
        print("      Error: {}".format(e))

print("\n" + "=" * 60)

print("\nTesting Lingua Language Detection...")
print("=" * 60)

if lingua:
    print("\nTesting English sentences with Lingua:")
    print("-" * 40)
    for i, sentence in enumerate(english_sentences, 1):
        try:
            result = lingua_detector.detect_language_of(sentence)
            lang = result.iso_code_639_1.name.lower()
            confidence = lingua_detector.compute_language_confidence_values(sentence)[0].value
            print("EN {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Confidence: {:.3f}".format(lang, confidence))
        except Exception as e:
            print("EN {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Somali sentences with Lingua:")
    print("-" * 40)
    for i, sentence in enumerate(somali_sentences, 1):
        try:
            result = lingua_detector.detect_language_of(sentence)
            lang = result.iso_code_639_1.name.lower()
            confidence = lingua_detector.compute_language_confidence_values(sentence)[0].value
            print("SO {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Confidence: {:.3f}".format(lang, confidence))
        except Exception as e:
            print("SO {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Amharic sentences with Lingua:")
    print("-" * 40)
    for i, sentence in enumerate(amharic_sentences, 1):
        try:
            result = lingua_detector.detect_language_of(sentence)
            lang = result.iso_code_639_1.name.lower()
            confidence = lingua_detector.compute_language_confidence_values(sentence)[0].value
            print("AM {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Confidence: {:.3f}".format(lang, confidence))
        except Exception as e:
            print("AM {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Afrikaans sentences with Lingua:")
    print("-" * 40)
    for i, sentence in enumerate(afrikaans_sentences, 1):
        try:
            result = lingua_detector.detect_language_of(sentence)
            lang = result.iso_code_639_1.name.lower()
            confidence = lingua_detector.compute_language_confidence_values(sentence)[0].value
            print("AF {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Confidence: {:.3f}".format(lang, confidence))
        except Exception as e:
            print("AF {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Swahili sentences with Lingua:")
    print("-" * 40)
    for i, sentence in enumerate(swahili_sentences, 1):
        try:
            result = lingua_detector.detect_language_of(sentence)
            lang = result.iso_code_639_1.name.lower()
            confidence = lingua_detector.compute_language_confidence_values(sentence)[0].value
            print("SW {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Confidence: {:.3f}".format(lang, confidence))
        except Exception as e:
            print("SW {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Zulu sentences with Lingua:")
    print("-" * 40)
    for i, sentence in enumerate(zulu_sentences, 1):
        try:
            result = lingua_detector.detect_language_of(sentence)
            lang = result.iso_code_639_1.name.lower()
            confidence = lingua_detector.compute_language_confidence_values(sentence)[0].value
            print("ZU {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Confidence: {:.3f}".format(lang, confidence))
        except Exception as e:
            print("ZU {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Hausa sentences with Lingua:")
    print("-" * 40)
    for i, sentence in enumerate(hausa_sentences, 1):
        try:
            result = lingua_detector.detect_language_of(sentence)
            lang = result.iso_code_639_1.name.lower()
            confidence = lingua_detector.compute_language_confidence_values(sentence)[0].value
            print("HA {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Confidence: {:.3f}".format(lang, confidence))
        except Exception as e:
            print("HA {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))
else:
    print("⚠️ Lingua detector not loaded, skipping testing")

print("\n" + "=" * 60)


print("Testing AfroLM Language Detection...")
print("=" * 60)

if afrolm_model is not None and afrolm_tokenizer is not None:
    print("\nTesting English sentences with AfroLM:")
    print("-" * 40)
    for i, sentence in enumerate(english_sentences, 1):
        try:
            result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
            lang = result[0][0]
            score = result[0][1]
            print("EN {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Score: {:.3f}".format(lang, score))
        except Exception as e:
            print("EN {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Somali sentences with AfroLM:")
    print("-" * 40)
    for i, sentence in enumerate(somali_sentences, 1):
        try:
            result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
            lang = result[0][0]
            score = result[0][1]
            print("SO {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Score: {:.3f}".format(lang, score))
        except Exception as e:
            print("SO {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Amharic sentences with AfroLM:")
    print("-" * 40)
    for i, sentence in enumerate(amharic_sentences, 1):
        try:
            result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
            lang = result[0][0]
            score = result[0][1]
            print("AM {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Score: {:.3f}".format(lang, score))
        except Exception as e:
            print("AM {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Afrikaans sentences with AfroLM:")
    print("-" * 40)
    for i, sentence in enumerate(afrikaans_sentences, 1):
        try:
            result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
            lang = result[0][0]
            score = result[0][1]
            print("AF {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Score: {:.3f}".format(lang, score))
        except Exception as e:
            print("AF {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Swahili sentences with AfroLM:")
    print("-" * 40)
    for i, sentence in enumerate(swahili_sentences, 1):
        try:
            result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
            lang = result[0][0]
            score = result[0][1]
            print("SW {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Score: {:.3f}".format(lang, score))
        except Exception as e:
            print("SW {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Zulu sentences with AfroLM:")
    print("-" * 40)
    for i, sentence in enumerate(zulu_sentences, 1):
        try:
            result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
            lang = result[0][0]
            score = result[0][1]
            print("ZU {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Score: {:.3f}".format(lang, score))
        except Exception as e:
            print("ZU {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))

    print("\nTesting Hausa sentences with AfroLM:")
    print("-" * 40)
    for i, sentence in enumerate(hausa_sentences, 1):
        try:
            result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
            lang = result[0][0]
            score = result[0][1]
            print("HA {}: '{}'...".format(i, sentence[:50]))
            print("      Detected: {} | Score: {:.3f}".format(lang, score))
        except Exception as e:
            print("HA {}: '{}'...".format(i, sentence[:50]))
            print("      Error: {}".format(e))
else:
    print("⚠️ AfroLM model not loaded, skipping testing")

print("\n" + "=" * 60)



# Function to analyze language detection performance
def analyze_language_detection():
    results = {
            'LangDetect': {
                'correct': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0},
                'total': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0}
            },
            'FastText': {
                'correct': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0},
                'total': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0}
            },
            'AfroLID': {
                'correct': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0},
                'total': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0}
            },
            'PyFranc': {
                'correct': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0},
                'total': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0}
            },
            'Lingua': {
                'correct': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0},
                'total': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0}
            },
            'AfroLM': {
                'correct': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0},
                'total': {'en': 0, 'af': 0, 'so': 0, 'am': 0, 'sw': 0, 'zu': 0, 'ha': 0}
            }
        }
    
    # Language code mappings for each method
    code_mappings = {
        'LangDetect': {'en': 'en', 'af': 'af', 'so': 'so', 'am': 'am', 'sw': 'sw', 'zu': 'zu', 'ha': 'ha'},
        'FastText': {'en': 'en', 'af': 'afr', 'so': 'som', 'am': 'amh', 'sw': 'swa', 'zu': 'zul', 'ha': 'hau'},
        'AfroLID': {'en': 'en', 'af': 'afr', 'so': 'som', 'am': 'amh', 'sw': 'swa', 'zu': 'zul', 'ha': 'hau'},
        'PyFranc': {'en': 'eng', 'af': 'afr', 'so': 'som', 'am': 'amh', 'sw': 'swa', 'zu': 'zul', 'ha': 'hau'},
        'Lingua': {'en': 'en', 'af': 'af', 'so': 'so', 'am': 'am', 'sw': 'sw', 'zu': 'zu', 'ha': 'ha'},
        'AfroLM': {'en': 'en', 'af': 'af', 'so': 'so', 'am': 'am', 'sw': 'sw', 'zu': 'zu', 'ha': 'ha'}
    }

    # Test data
    test_data = [
        ('en', english_sentences), ('af', afrikaans_sentences), ('so', somali_sentences),
        ('am', amharic_sentences), ('sw', swahili_sentences), ('zu', zulu_sentences), ('ha', hausa_sentences)
    ]

    # LangDetect analysis
    for lang_code, sentences in test_data:
        for sentence in sentences:
            try:
                predicted = detect(sentence)
                results['LangDetect']['total'][lang_code] += 1
                if predicted == code_mappings['LangDetect'][lang_code]:
                    results['LangDetect']['correct'][lang_code] += 1
            except:
                results['LangDetect']['total'][lang_code] += 1

    # FastText analysis
    try:
        for lang_code, sentences in test_data:
            for sentence in sentences:
                try:
                    predictions = model.predict(sentence, k=1)
                    predicted = predictions[0][0].replace("__label__", "")
                    results['FastText']['total'][lang_code] += 1
                    if predicted == code_mappings['FastText'][lang_code]:
                        results['FastText']['correct'][lang_code] += 1
                except:
                    results['FastText']['total'][lang_code] += 1
    except:
        pass  # FastText not available

    # AfroLID analysis
    try:
        for lang_code, sentences in test_data:
            for sentence in sentences:
                try:
                    result = afrolid(sentence)
                    predicted = result[0]['label']
                    results['AfroLID']['total'][lang_code] += 1
                    if predicted == code_mappings['AfroLID'][lang_code]:
                        results['AfroLID']['correct'][lang_code] += 1
                except:
                    results['AfroLID']['total'][lang_code] += 1
    except:
        pass  # AfroLID not available

    # PyFranc analysis
    for lang_code, sentences in test_data:
        for sentence in sentences:
            try:
                result = franc.lang_detect(sentence)
                predicted = result[0][0]
                results['PyFranc']['total'][lang_code] += 1
                if predicted == code_mappings['PyFranc'][lang_code]:
                    results['PyFranc']['correct'][lang_code] += 1
            except:
                results['PyFranc']['total'][lang_code] += 1

    # Lingua analysis
    if lingua:
        for lang_code, sentences in test_data:
            for sentence in sentences:
                try:
                    result = lingua_detector.detect_language_of(sentence)
                    predicted = result.iso_code_639_1.name.lower()
                    results['Lingua']['total'][lang_code] += 1
                    if predicted == code_mappings['Lingua'][lang_code]:
                        results['Lingua']['correct'][lang_code] += 1
                except:
                    results['Lingua']['total'][lang_code] += 1
    else:
        print("⚠️ Lingua detector not loaded, skipping lingua analysis")

    # AfroLM analysis
    if afrolm_model is not None and afrolm_tokenizer is not None:
        for lang_code, sentences in test_data:
            for sentence in sentences:
                try:
                    result = afrolm_language_detect(sentence, afrolm_model, afrolm_tokenizer)
                    predicted = result[0][0]
                    results['AfroLM']['total'][lang_code] += 1
                    if predicted == code_mappings['AfroLM'][lang_code]:
                        results['AfroLM']['correct'][lang_code] += 1
                except:
                    results['AfroLM']['total'][lang_code] += 1
    else:
        print("⚠️ AfroLM model not loaded, skipping AfroLM analysis")

    return results, test_data
# Run analysis
results, test_data = analyze_language_detection()

print("\n" + "=" * 100)
print("📊 NUMERICAL ANALYSIS: Language Detection Accuracy Comparison")
print("=" * 100)

# Function to display results in table format
def display_results_table(results, title="Language Detection Accuracy by Method"):
    print(f"\n{title}")
    print("=" * 80)

    # Table header
    header = "| {:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |".format(
        "Method", "EN", "AF", "SO", "AM", "SW", "ZU", "HA", "TOTAL"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    language_names = {'en': 'English', 'af': 'Afrikaans', 'so': 'Somali',
                     'am': 'Amharic', 'sw': 'Swahili', 'zu': 'Zulu', 'ha': 'Hausa'}

    for method in ['LangDetect', 'FastText', 'AfroLID', 'PyFranc', 'Lingua', 'AfroLM']:
        if method in results:
            row_data = []
            total_correct = 0
            total_sentences = 0

            for lang in ['en', 'af', 'so', 'am', 'sw', 'zu', 'ha']:
                correct = results[method]['correct'][lang]
                total = results[method]['total'][lang]
                total_correct += correct
                total_sentences += total

                if total > 0:
                    accuracy = (correct / total) * 100
                    accuracy_str = f"{accuracy:6.1f}%"
                    # Simple color coding
                    if accuracy >= 90:
                        row_data.append(f"{accuracy_str}***")  # *** for excellent
                    elif accuracy >= 70:
                        row_data.append(f"{accuracy_str}** ")  # ** for good
                    else:
                        row_data.append(f"{accuracy_str}*  ")  # * for needs improvement
                else:
                    row_data.append("   N/A ")

            # Overall accuracy
            if total_sentences > 0:
                overall_accuracy = (total_correct / total_sentences) * 100
                overall_str = f"{overall_accuracy:5.1f}%"
            else:
                overall_str = " N/A "

            row = "| {:<10} | {} | {} | {} | {} | {} | {} | {} | {:<7} |".format(
                method, row_data[0], row_data[1], row_data[2], row_data[3],
                row_data[4], row_data[5], row_data[6], overall_str
            )
            print(row)

    print("-" * len(header))

    # Add color legend
    print("\n📈 Legend:")
    print("*** >= 90% : Excellent     ** >= 70% : Good     * < 70% : Needs Improvement")

# Display the main results table
display_results_table(results)

print("\n" + "=" * 100)
print("🏆 DETAILED PERFORMANCE ANALYSIS BY LANGUAGE")
print("=" * 100)

# Function to show per-method analysis
lang_codes = ['en', 'af', 'so', 'am', 'sw', 'zu', 'ha']
lang_names = ['English', 'Afrikaans', 'Somali', 'Amharic', 'Swahili', 'Zulu', 'Hausa']

print("\n1️⃣ LANGDETECT PERFORMANCE BREAKDOWN:")
print("-" * 50)
for i, (code, lang) in enumerate(zip(lang_codes, lang_names)):
    if code in results['LangDetect']['total']:
        correct = results['LangDetect']['correct'][code]
        total = results['LangDetect']['total'][code]
        if total > 0:
            acc = (correct / total) * 100
            print(f"{lang}: {correct:2d}/{total} correct ({acc:5.1f}%)")

print("\n2️⃣ FASTTEXT PERFORMANCE BREAKDOWN:")
print("-" * 50)
for i, (code, lang) in enumerate(zip(lang_codes, lang_names)):
    if code in results['FastText']['total']:
        correct = results['FastText']['correct'][code]
        total = results['FastText']['total'][code]
        if total > 0:
            acc = (correct / total) * 100
            print(f"{lang}: {correct:2d}/{total} correct ({acc:5.1f}%)")

print("\n3️⃣ AFROLID PERFORMANCE BREAKDOWN:")
print("-" * 50)
for i, (code, lang) in enumerate(zip(lang_codes, lang_names)):
    if code in results['AfroLID']['total']:
        correct = results['AfroLID']['correct'][code]
        total = results['AfroLID']['total'][code]
        if total > 0:
            acc = (correct / total) * 100
            print(f"{lang}: {correct:2d}/{total} correct ({acc:5.1f}%)")

print("\n4️⃣ PYFRANC PERFORMANCE BREAKDOWN:")
print("-" * 50)
for i, (code, lang) in enumerate(zip(lang_codes, lang_names)):
    if code in results['PyFranc']['total']:
        correct = results['PyFranc']['correct'][code]
        total = results['PyFranc']['total'][code]
        if total > 0:
            acc = (correct / total) * 100
            print(f"{lang}: {correct:2d}/{total} correct ({acc:5.1f}%)")

print("\n5️⃣ LINGUA PERFORMANCE BREAKDOWN:")
print("-" * 50)
for i, (code, lang) in enumerate(zip(lang_codes, lang_names)):
    if code in results['Lingua']['total']:
        correct = results['Lingua']['correct'][code]
        total = results['Lingua']['total'][code]
        if total > 0:
            acc = (correct / total) * 100
            print(f"{lang}: {correct:2d}/{total} correct ({acc:5.1f}%)")

print("\n6️⃣ AFROLM PERFORMANCE BREAKDOWN:")
print("-" * 50)
for i, (code, lang) in enumerate(zip(lang_codes, lang_names)):
    if code in results['AfroLM']['total']:
        correct = results['AfroLM']['correct'][code]
        total = results['AfroLM']['total'][code]
        if total > 0:
            acc = (correct / total) * 100
            print(f"{lang}: {correct:2d}/{total} correct ({acc:5.1f}%)")

print("\n" + "=" * 100)
print("🏆 WINNER BY LANGUAGE CATEGORY")
print("=" * 100)

# Calculate winners for each language
print("\n📊 Best Method for Each Language:")
print("-" * 50)

winners = {}
for code, lang in zip(lang_codes, lang_names):
    method_scores = {}

    for method in ['LangDetect', 'FastText', 'AfroLID', 'PyFranc', 'Lingua', 'AfroLM']:
        if code in results[method]['total']:
            total = results[method]['total'][code]
            if total > 0:
                correct = results[method]['correct'][code]
                accuracy = (correct / total) * 100
                method_scores[method] = accuracy

    if method_scores:
        winner_method = max(method_scores, key=method_scores.get)
        winner_score = method_scores[winner_method]
        print(f"{lang:<12s}: {winner_method:8s} ({winner_score:5.1f}%)")

print("\n" + "=" * 100)
print("🥇 FINAL RECOMMENDATION")
print("=" * 100)

# Overall performance calculation
overall_scores = {}
for method in ['LangDetect', 'FastText', 'AfroLID', 'PyFranc', 'Lingua', 'AfroLM']:
    total_correct = 0
    total_sentences = 0
    for code in lang_codes:
        if code in results[method]['total']:
            total_correct += results[method]['correct'][code]
            total_sentences += results[method]['total'][code]

    if total_sentences > 0:
        overall_scores[method] = (total_correct / total_sentences) * 100

if overall_scores:
    best_method = max(overall_scores, key=overall_scores.get)
    best_score = overall_scores[best_method]

    print(f"🎯 OVERALL ACCURACY SCORES:")
    for method, score in overall_scores.items():
        print(f"   {method:<12s}: {score:5.1f}%")

    print(f"\n🏆 BEST METHOD: {best_method}")
    print(f"🏆 ACCURACY: {best_score:.1f}%")
    print(f"\n💡 RECOMMENDATION: Use {best_method} for your African language dataset!")
else:
    print("❌ No data available for comparison")
