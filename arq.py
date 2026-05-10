# =========================================================
# NLP INDUSTRIAL MDM
# CLASSIFICADOR + NLP INDUSTRIAL
# =========================================================

import os
import re
import json
import time
import unicodedata

from collections import Counter
from collections import defaultdict

import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "input"
OUTPUT_DIR = "output"
CONFIG_DIR = "config"

MAX_LINHAS = 100000

MAX_SUGESTOES = 100
MIN_FREQ_SUGESTAO = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPO_INICIO = time.time()


# =========================================================
# LOG
# =========================================================

def log(msg):

    tempo = round(
        time.time() - TEMPO_INICIO,
        2
    )

    print(f"[{tempo}s] {msg}")


# =========================================================
# LOAD JSON
# =========================================================

def load_json(nome):

    path = os.path.join(
        CONFIG_DIR,
        nome
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


taxonomy = load_json("taxonomy.json")

modifiers_cfg = load_json(
    "modifiers.json"
)

synonyms = load_json(
    "synonyms.json"
)

compound_nouns_cfg = load_json(
    "compound_nouns.json"
)

oem_cfg = load_json(
    "oem.json"
)

stopwords = set(
    load_json("stopwords.json")
)

blocked_nouns = set(
    load_json("blocked_nouns.json")
)

material_words = set(
    load_json("material_words.json")
)

application_words = set(
    load_json("application_words.json")
)

norma_words = set(
    load_json("norma_words.json")
)

cores = set(
    load_json("cores.json")
)

dimensional_words = set(
    load_json("dimensional_words.json")
)

measurement_words = set(
    load_json("measurement_words.json")
)

attribute_words = set(
    load_json("attribute_words.json")
)

documental_words = set(
    load_json("documental_words.json")
)

protecao_words = set(
    load_json("protecao_words.json")
)

propulsao_words = set(
    load_json("propulsao_words.json")
)

technical_short_tokens = set(
    load_json(
        "technical_short_tokens.json"
    )
)



# =========================================================
# EMBEDDING
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

categoria_textos = {}
centroides = {}


# =========================================================
# REGEX
# =========================================================

NUMERO_REGEX = re.compile(
    r"^\d+([.,]\d+)?$"
)

FRACAO_REGEX = re.compile(
    r"^\d+/\d+$"
)

UNIDADE_REGEX = re.compile(
    r"^\d+([.,]\d+)?(MM|CM|POL|KG|G|L|BAR|PSI|V|KV|HZ|RPM)$"
)

ANGULO_REGEX = re.compile(
    r"^\d+(GR|GRAUS)$"
)

CODIGO_REGEX = re.compile(
    r"^[A-Z0-9./_-]{8,}$"
)


# =========================================================
# NORMALIZE
# =========================================================

def remover_acentos(txt):

    txt = unicodedata.normalize(
        "NFKD",
        txt
    )

    return "".join(

        c for c in txt

        if not unicodedata.combining(c)
    )


def normalize(txt):

    if txt is None:
        return ""

    txt = str(txt)

    txt = remover_acentos(txt)

    txt = txt.upper()

    txt = txt.replace(
        "_X0000_",
        " "
    )

    txt = txt.replace(
        "X0000",
        " "
    )

    txt = txt.replace(
        "ACOO",
        "ACO"
    )

    txt = txt.replace(
        "INOXX",
        "INOX"
    )

    txt = re.sub(
        r"(\d),(\d)",
        r"\1.\2",
        txt
    )

    txt = re.sub(
        r"[^A-Z0-9./%+\-_]+",
        " ",
        txt
    )

    txt = re.sub(
        r"\s+",
        " ",
        txt
    )

    return txt.strip()


# Definitions that depend on normalize()
OEM_SET = set([

    normalize_key.upper()

    for normalize_key in oem_cfg.keys()
])

COMPOUND_NOUNS = {

    tuple(normalize(k).split()):
        normalize(k).replace(" ", "_")

    for k in compound_nouns_cfg.keys()
}

COMPOUND_NOUNS.update({
    ("O", "RING"): "O-RING",
    ("BLADDER", "KIT"): "BLADDER_KIT"
})

MAX_COMPOUND = max(
    (len(k) for k in COMPOUND_NOUNS),
    default=2
)

LEXICOS_BLOQUEADOS = (
    blocked_nouns
    | material_words
    | norma_words
    | cores
    | dimensional_words
    | measurement_words
    | attribute_words
    | documental_words
    | protecao_words
    | propulsao_words
    | technical_short_tokens
    | OEM_SET
)


# =========================================================
# SYNONYMS
# =========================================================

def aplicar_synonyms(texto):

    texto = normalize(texto)

    for k, v in synonyms.items():

        texto = texto.replace(

            normalize(k),
            normalize(v)
        )

    return texto


# =========================================================
# MODIFIERS
# =========================================================

def aplicar_modifiers_semanticos(
    texto
):

    texto = normalize(texto)

    tokens = texto.split()

    out = []

    i = 0

    while i < len(tokens):

        atual = tokens[i]

        prox = None

        if i + 1 < len(tokens):

            prox = tokens[i + 1]

        substituiu = False

        if atual in modifiers_cfg and prox:

            mods = modifiers_cfg[atual]

            if isinstance(mods, dict):

                if prox in mods:

                    canonico = mods[prox]

                    out.append(

                        canonico.replace(
                            " ",
                            "_"
                        )
                    )

                    i += 2

                    substituiu = True

        if (
            not substituiu
            and prox
        ):

            if prox in modifiers_cfg:

                mods = modifiers_cfg[prox]

                if isinstance(mods, dict):

                    if atual in mods:

                        canonico = mods[atual]

                        out.append(

                            canonico.replace(
                                " ",
                                "_"
                            )
                        )

                        i += 2

                        substituiu = True

        if not substituiu:

            out.append(atual)

            i += 1

    return " ".join(out)


# =========================================================
# TOKEN INVALIDO
# =========================================================

def token_ruim(t):

    if not t:
        return True

    partes = [x for x in re.split(r"[ _-]+", t) if x]

    if t in LEXICOS_BLOQUEADOS:
        return True

    if any(parte in LEXICOS_BLOQUEADOS for parte in partes):
        return True

    if re.match(r"^CAT\d", t):
        return True

    if re.match(r"^[MT]\d{1,3}$", t):
        return True

    if re.match(r"^CAS$", t):
        return True

    if re.match(r"^EXT[A-Z]$", t):
        return True

    if re.match(r"^RSCD$", t):
        return True

    if NUMERO_REGEX.match(t):
        return True

    if FRACAO_REGEX.match(t):
        return True

    if UNIDADE_REGEX.match(t):
        return True

    if ANGULO_REGEX.match(t):
        return True

    if CODIGO_REGEX.match(t):
        return True

    if t.endswith("MM"):
        return True

    if t.endswith("POL"):
        return True

    return False


# =========================================================
# TOKENIZE
# =========================================================

def tokenize(texto):

    try:

        texto = aplicar_synonyms(
            texto
        )

        texto = aplicar_modifiers_semanticos(
            texto
        )

        if not isinstance(texto, str):

            return []

        tokens = texto.split()

        out = []

        i = 0

        while i < len(tokens):

            matched = False

            for n in range(
                min(MAX_COMPOUND, len(tokens) - i),
                1,
                -1
            ):

                gram = tuple(tokens[i:i+n])

                if gram in COMPOUND_NOUNS:

                    out.append(COMPOUND_NOUNS[gram])

                    i += n

                    matched = True
                    break

            if matched:
                continue

            t = tokens[i]

            t = t.replace(
                "_",
                " "
            )

            if len(t) <= 1:

                i += 1
                continue

            if t in stopwords:

                i += 1
                continue

            out.append(t)

            i += 1

        return out

    except Exception as e:

        print(
            f"ERRO TOKENIZE: {e}"
        )

        return []


# =========================================================
# ENTITY EXTRACTION
# =========================================================

def extrair_entidades(tokens):

    entidades = {

        "NOUN": [],
        "MATERIAL": [],
        "APPLICATION": [],
        "OEM": [],
        "NORMA": [],
        "COR": []
    }

    for t in tokens:

        if t in material_words:

            entidades["MATERIAL"].append(
                t
            )

        elif t in application_words:

            entidades["APPLICATION"].append(
                t
            )

        elif t in OEM_SET:

            entidades["OEM"].append(
                t
            )

        elif t in norma_words:

            entidades["NORMA"].append(
                t
            )

        elif t in cores:

            entidades["COR"].append(
                t
            )

        elif not token_ruim(t):

            entidades["NOUN"].append(
                t
            )

    return entidades


# =========================================================
# EMBEDDING
# =========================================================

def adicionar_categoria(cat, texto):

    if cat not in categoria_textos:

        categoria_textos[cat] = []

    if len(categoria_textos[cat]) < 50:

        categoria_textos[cat].append(
            texto
        )


def construir_centroides():

    for cat, textos in categoria_textos.items():

        if textos:

            emb = embedding_model.encode(
                textos
            )

            centroides[cat] = emb.mean(
                axis=0
            )


def classificar_embedding(texto):

    if not centroides:
        return None, 0

    emb = embedding_model.encode(
        [texto]
    )[0]

    melhor = None
    score_max = 0

    for cat, centroide in centroides.items():

        score = cosine_similarity(

            [emb],
            [centroide]

        )[0][0]

        if score > score_max:

            melhor = cat
            score_max = score

    if score_max < 0.45:

        return None, score_max

    return melhor, score_max


# =========================================================
# CLASSIFICADOR
# =========================================================

def classificar_item(texto, codigo=""):

    codigo_n = normalize(codigo)

    # =====================================================
    # SERVICO
    # =====================================================

    if codigo_n.startswith("S0"):

        return {

            "categoria": "SERVICO",
            "familia": "SERVICO",
            "confianca": "ALTA",
            "servico": True
        }

    # =====================================================
    # RANCHO
    # =====================================================

    if codigo_n.startswith("RN"):

        return {

            "categoria": "RANCHO",
            "familia": "RANCHO",
            "confianca": "ALTA",
            "servico": False
        }

    texto_n = normalize(texto)

    tokens = tokenize(texto_n)

    if not isinstance(tokens, list):

        tokens = []

    if not tokens:

        return {

            "categoria":
                "NAO_CLASSIFICADO",

            "familia":
                "NAO_CLASSIFICADO",

            "confianca":
                "BAIXA",

            "servico": False
        }

    scores = defaultdict(int)

    # =====================================================
    # TAXONOMY
    # =====================================================

    for idx, t in enumerate(tokens):

        if t in taxonomy:

            cat = taxonomy[t]

            scores[cat] += 1

    # =====================================================
    # RESULTADO
    # =====================================================

    if scores:

        melhor = max(

            scores.items(),
            key=lambda x: x[1]

        )[0]

        return {

            "categoria": melhor,
            "familia": melhor,
            "confianca": "ALTA",
            "servico": False
        }

    # =====================================================
    # EMBEDDING
    # =====================================================

    emb_cat, score = classificar_embedding(
        texto_n
    )

    if emb_cat:

        return {

            "categoria": emb_cat,
            "familia": emb_cat,
            "confianca": "MEDIA",
            "score": round(score, 3),
            "servico": False
        }

    return {

        "categoria": "NAO_CLASSIFICADO",
        "familia": "NAO_CLASSIFICADO",
        "confianca": "BAIXA",
        "servico": False
    }


# =========================================================
# SUGESTOES
# =========================================================

contador_nouns = Counter()
contador_modifiers = Counter()
contador_oem = Counter()
contador_application = Counter()
contador_categoria = Counter()

exemplos_nouns = {}
exemplos_modifiers = {}
exemplos_oem = {}
exemplos_application = {}


def registrar_sugestoes(texto):

    tokens = tokenize(texto)

    entidades = extrair_entidades(
        tokens
    )

    # =====================================================
    # NOUNS
    # =====================================================

    nouns_unicos = set(
        entidades["NOUN"]
    )

    for t in nouns_unicos:

        contador_nouns[t] += 1

        exemplos_nouns.setdefault(
            t,
            []
        ).append(texto[:300])

    # =====================================================
    # OEM
    # =====================================================

    for t in set(entidades["OEM"]):

        contador_oem[t] += 1

        exemplos_oem.setdefault(
            t,
            []
        ).append(texto[:300])

    # =====================================================
    # APPLICATION
    # =====================================================

    for t in set(entidades["APPLICATION"]):

        contador_application[t] += 1

        exemplos_application.setdefault(
            t,
            []
        ).append(texto[:300])

    # =====================================================
    # MODIFIERS
    # =====================================================

    MAX_NGRAM = 3

    for n in range(2, MAX_NGRAM + 1):

        vistos = set()

        for i in range(len(tokens)-n+1):

            termos = tokens[i:i+n]

            if any(
                token_ruim(x)
                for x in termos
            ):
                continue

            chave = tuple(termos)

            if chave in vistos:
                continue

            vistos.add(chave)

            contador_modifiers[
                chave
            ] += 1

            exemplos_modifiers.setdefault(
                chave,
                []
            ).append(texto[:300])


# =========================================================
# EXPORT
# =========================================================

def exportar_counter(

    counter,
    exemplos,
    nome,
    campo
):

    rows = []

    for termo, qtd in counter.most_common():

        if qtd < MIN_FREQ_SUGESTAO:
            continue

        exemplo = ""

        if termo in exemplos:

            exemplo = exemplos[
                termo
            ][0][:300]

        if isinstance(termo, tuple):

            termo = " | ".join(termo)

        rows.append({

            campo: termo,
            "QTD": int(qtd),
            "EXEMPLO": exemplo
        })

    df = pd.DataFrame(rows)

    df = df.head(MAX_SUGESTOES)

    path = os.path.join(
        OUTPUT_DIR,
        nome
    )

    df.to_excel(
        path,
        index=False
    )

    log(f"Arquivo exportado: {nome}")


# =========================================================
# EXPORT CATEGORIAS
# =========================================================

def exportar_categorias():

    rows = []

    for cat, qtd in contador_categoria.most_common():

        rows.append({

            "CATEGORIA": cat,
            "TOTAL_ITENS": int(qtd)
        })

    df = pd.DataFrame(rows)

    path = os.path.join(

        OUTPUT_DIR,
        "total_categoria.xlsx"
    )

    df.to_excel(
        path,
        index=False
    )

    log(
        "Arquivo exportado: total_categoria.xlsx"
    )


# =========================================================
# COLUNAS
# =========================================================

def mapear_colunas(df):

    cols = list(df.columns)

    return {

        "CODIGO":
            cols[0] if len(cols) > 0 else None,

        "DESCRICAO":
            cols[1] if len(cols) > 1 else None,

        "LONGA":
            cols[2] if len(cols) > 2 else None,

        "PART":
            cols[3] if len(cols) > 3 else None
    }


# =========================================================
# PROCESSAMENTO
# =========================================================

arquivos = [

    f for f in os.listdir(INPUT_DIR)

    if f.endswith(
        (
            ".xlsx",
            ".xls",
            ".csv"
        )
    )
]

resultado_total = []

log(f"Arquivos encontrados: {len(arquivos)}")


for arq in arquivos:

    path = os.path.join(
        INPUT_DIR,
        arq
    )

    log(f"Processando {arq}")

    # =====================================================
    # LOAD
    # =====================================================

    if arq.endswith(".csv"):

        df = pd.read_csv(
            path,
            dtype=str
        )

    else:

        df = pd.read_excel(
            path,
            dtype=str
        )

    df = df.fillna("")

    df = df.head(MAX_LINHAS)

    df.columns = [

        normalize(c)

        for c in df.columns
    ]

    cols = mapear_colunas(df)

    resultados = []

    # =====================================================
    # LOOP
    # =====================================================

    for _, row in df.iterrows():

        codigo = str(

            row.get(
                cols["CODIGO"],
                ""
            )
        )

        descricao = str(

            row.get(
                cols["DESCRICAO"],
                ""
            )
        )

        longa = str(

            row.get(
                cols["LONGA"],
                ""
            )
        )

        part = str(

            row.get(
                cols["PART"],
                ""
            )
        )

        texto = (

            descricao
            + " "
            + longa
            + " "
            + part
        )

        # =================================================
        # CLASSIFICACAO
        # =================================================

        classif = classificar_item(
            texto,
            codigo
        )

        contador_categoria[
            classif["categoria"]
        ] += 1

        # =================================================
        # SERVICO
        # =================================================

        if classif["servico"]:

            resultados.append({

                **row.to_dict(),

                "CODIGO": codigo,

                "CATEGORIA":
                    "SERVICO",

                "FAMILIA":
                    "SERVICO",

                "CONFIANCA":
                    "ALTA"
            })

            continue

        # =================================================
        # NLP
        # =================================================

        registrar_sugestoes(texto)

        if (
            classif["categoria"]
            != "NAO_CLASSIFICADO"
        ):

            adicionar_categoria(

                classif["categoria"],
                texto
            )

        entidades = extrair_entidades(

            tokenize(texto)
        )

        resultados.append({

            **row.to_dict(),

            "CODIGO": codigo,

            "CATEGORIA":
                classif["categoria"],

            "FAMILIA":
                classif["familia"],

            "CONFIANCA":
                classif["confianca"],

            "NOUNS":
                " | ".join(
                    entidades["NOUN"]
                ),

            "MATERIAL":
                " | ".join(
                    entidades["MATERIAL"]
                ),

            "APPLICATION":
                " | ".join(
                    entidades["APPLICATION"]
                ),

            "OEM":
                " | ".join(
                    entidades["OEM"]
                ),

            "NORMA":
                " | ".join(
                    entidades["NORMA"]
                )
        })

    # =====================================================
    # EXPORT
    # =====================================================

    df_out = pd.DataFrame(resultados)

    out_path = os.path.join(

        OUTPUT_DIR,

        f"resultado_{arq}.xlsx"
    )

    df_out.to_excel(
        out_path,
        index=False
    )

    resultado_total.append(df_out)

    log(f"{arq} finalizado")


# =========================================================
# EMBEDDING
# =========================================================

construir_centroides()


# =========================================================
# CONSOLIDADO
# =========================================================

if resultado_total:

    pd.concat(
        resultado_total
    ).to_excel(

        os.path.join(
            OUTPUT_DIR,
            "resultado_total.xlsx"
        ),

        index=False
    )


# =========================================================
# EXPORT SUGESTOES
# =========================================================

exportar_counter(

    contador_nouns,
    exemplos_nouns,
    "sugestao_nouns.xlsx",
    "NOUN"
)

exportar_counter(

    contador_modifiers,
    exemplos_modifiers,
    "sugestao_modifiers.xlsx",
    "TERMOS"
)

exportar_counter(

    contador_oem,
    exemplos_oem,
    "sugestao_oem.xlsx",
    "OEM"
)

exportar_counter(

    contador_application,
    exemplos_application,
    "sugestao_application.xlsx",
    "APPLICATION"
)

exportar_categorias()

# =========================================================
# FINAL
# =========================================================

log("PROCESSAMENTO FINALIZADO")