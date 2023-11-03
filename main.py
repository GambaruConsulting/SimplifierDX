import logging
import onnxruntime
import re
import json
device = 'cpu'
import os
from transformers import T5TokenizerFast, pipeline

#region Load Model and Tokenizer
from fastT5 import (OnnxT5, get_onnx_runtime_sessions)

trained_model_path = "./onnx"
model_name = "model_files"


options = onnxruntime.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 1
#options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL


#ClONED BELOW
encoder_path = os.path.join(trained_model_path, f"{model_name}-encoder-quantized.onnx")
decoder_path = os.path.join(trained_model_path, f"{model_name}-decoder-quantized.onnx")
init_decoder_path = os.path.join(trained_model_path, f"{model_name}-init-decoder-quantized.onnx")


#MERGE
with open(os.path.join(trained_model_path, f"encoder_0.onnx"), "rb") as file_0, \
        open(os.path.join(trained_model_path, f"encoder_1.onnx"), "rb") as file_1, \
        open(os.path.join(trained_model_path, f"encoder_2.onnx"), "rb") as file_2, \
        open(os.path.join(trained_model_path, f"encoder_3.onnx"), "rb") as file_3, \
        open(os.path.join(trained_model_path, f"encoder_4.onnx"), "rb") as file_4:
    with open(encoder_path, "wb") as write_file:
        write_file.write(file_0.read() + file_1.read() + file_2.read() + file_3.read() + file_4.read())


with open(os.path.join(trained_model_path, f"decoder_0.onnx"), "rb") as file_0, \
        open(os.path.join(trained_model_path, f"decoder_1.onnx"), "rb") as file_1, \
        open(os.path.join(trained_model_path, f"decoder_2.onnx"), "rb") as file_2, \
        open(os.path.join(trained_model_path, f"decoder_3.onnx"), "rb") as file_3, \
        open(os.path.join(trained_model_path, f"decoder_4.onnx"), "rb") as file_4, \
        open(os.path.join(trained_model_path, f"decoder_5.onnx"), "rb") as file_5:
    with open(decoder_path, "wb") as write_file:
        write_file.write(file_0.read() + file_1.read() + file_2.read() + file_3.read() + file_4.read() + file_5.read())


with open(os.path.join(trained_model_path, f"init_decoder_0.onnx"), "rb") as file_0, \
        open(os.path.join(trained_model_path, f"init_decoder_1.onnx"), "rb") as file_1, \
        open(os.path.join(trained_model_path, f"init_decoder_2.onnx"), "rb") as file_2, \
        open(os.path.join(trained_model_path, f"init_decoder_3.onnx"), "rb") as file_3, \
        open(os.path.join(trained_model_path, f"init_decoder_4.onnx"), "rb") as file_4, \
        open(os.path.join(trained_model_path, f"init_decoder_5.onnx"), "rb") as file_5, \
        open(os.path.join(trained_model_path, f"init_decoder_6.onnx"), "rb") as file_6:
    with open(init_decoder_path, "wb") as write_file:
        write_file.write(file_0.read() + file_1.read() + file_2.read() + file_3.read() + file_4.read() + file_5.read() + file_6.read())
#end MERGE


#model_paths = encoder_path, decoder_path, init_decoder_path
#model_sessions = get_onnx_runtime_sessions(model_paths)

encoder_sess = onnxruntime.InferenceSession(str(encoder_path), sess_options=options)
decoder_sess = onnxruntime.InferenceSession(str(decoder_path), sess_options=options)
decoder_sess_init = onnxruntime.InferenceSession(str(init_decoder_path), sess_options=options)

model_sessions = encoder_sess, decoder_sess, decoder_sess_init

model = OnnxT5(trained_model_path, model_sessions)
tokenizer = T5TokenizerFast.from_pretrained(trained_model_path)
#endregion

#region Translation Functions
def SimplifyGate(input_string):
    if len(input_string) > 420:
        batchArray = []
        parts = splitSentence(input_string)
        for part in parts:
            batchArray.append(simplify(part))
        result = ""
        for item in batchArray:
            result = result + item + " "
    else:
        result = simplify(input_string)
    return result


def simplify(input_string, **generator_args):
    generator_args = {
        "num_beams": 5,
        "length_penalty": 1,
        "no_repeat_ngram_size": 5,
        "early_stopping": True,
        "min_length": 1,
        "max_length": 500
    }
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    output = [item.split("<sep>") for item in output][0][0]
    output = output[:1].upper() + output[1:]
    return output

"""generator_args = {
    "num_beams": 4,
    "length_penalty": 1,
    "no_repeat_ngram_size": 5,
    "early_stopping": True,
    "min_length": 1,
    "max_length": 500
}
simplify = pipeline(model=model, tokenizer=tokenizer, task="text2text-generation", **generator_args)[0]"""


pests = ["\nadvertisement"]


def splitSentence(sentence):
    sentences = []
    periods = [m.start() for m in re.finditer('\.\s', sentence)]
    abbreviations = [m.end() for m in re.finditer('(?:[A-Z][a-zA-Z]{0,}\.){1,}', sentence)]
    sentence_ends = periods
    for index in range(len(sentence_ends)):
        for item in abbreviations:
            if sentence_ends[index] + 1 == item:
                sentence_ends[index] = 0
    if sentence_ends.count(0) > 0:
        while 0 in sentence_ends: sentence_ends.remove(0)
    if len(sentence_ends) > 0:
        sentence_ends.insert(0, -2)
        sentence_ends.append(len(sentence))
        for index in range(len(sentence_ends)-1):
            sentences.append(sentence[sentence_ends[index]+2:sentence_ends[index+1]+1])
        return sentences
    else:
        return [sentence]


def cleanup(original):
    original = original.replace("’", "'")
    original = original.replace("—", "-")
    original = original.replace("“", '"')
    original = original.replace("”", '"')
    original = original.replace("‘", "'")
    original = original.replace("…", "...")
    original = original.replace("–", "-")
    original = original.replace("×", "x")
    original = original.replace("−", "-")
    original = original.replace("ä", "a")
    original = original.replace("å", "a")
    original = original.replace("á", "a")
    original = original.replace("à", "a")
    original = original.replace("æ", "ae")
    original = original.replace("ç", "c")
    original = original.replace("é", "e")
    original = original.replace("ë", "e")
    original = original.replace("í", "i")
    original = original.replace("ï", "i")
    original = original.replace("ł", "l")
    original = original.replace("ñ", "n")
    original = original.replace("ó", "o")
    original = original.replace("ö", "o")
    original = original.replace("ø", "o")
    original = original.replace("ü", "u")
    original = original.replace("£", "L")
    original = original.replace("©", "Copyright")
    original = original.replace("™", "(TM)")
    original = original.replace("®", "(registered)")
    original = original.replace("°", " degrees ")


    print(original)
    return original
#endregion

#region Postprocessing
def capitalize(sentence):
    if len(sentence) < 2:
        return sentence
    return sentence[:1].upper() + sentence[1:]

def uncapitalize(sentence):
    if len(sentence) < 2:
        return sentence
    return sentence[:1].lower() + sentence[1:]

def firstword(sentence):
    if " " in sentence:
        return sentence[:sentence.find(" ")]
    return sentence


async def PostProcess(input, original_output):

    if firstword(original_output) == firstword(input):
        return original_output

    # Solution 1:  Loop list of possibly cut things
    starters = ["The", "In", "In the", "At", "At the", "On", "On the", "This",
                "the", "in", "in the", "at", "at the", "on", "on the", "this",
                "Dr.", "Mr.", "Mrs.", "Ms."]

    first_word = firstword(original_output)

    if firstword(original_output) != ",":
        first_word = firstword(original_output).replace(",","")

    for starter in starters:
        if " " + starter + " " + first_word in input or input.find(starter + " " + first_word) == 0:
            new_output = starter + " " + original_output
            return capitalize(new_output)
        if " " + starter + " " + uncapitalize(first_word) in input or input.find(starter + " " + uncapitalize(first_word)) == 0:
            new_output = starter + " " + uncapitalize(original_output)
            return capitalize(new_output)

    # Solution 4:  Contractions
    def GetPronounFromContraction(contraction):
        return contraction[:contraction.find("\'")]

    contractions_i = ["I'm", "I've", "I'd", "I'll"]
    contractions_you = ["You're", "You've", "You'd", "You'll"]
    contractions_we = ["We're", "We've", "We'd", "We'll"]
    contractions_he = ["He's", "He'd", "He'll"]
    contractions_she = ["She's", "She'd", "She'll"]
    contractions_it = ["It's", "It'd", "It'll"]
    contractions_they = ["They're", "They've", "They'd", "They'll"]

    pronouns_that_take_contractions = [contractions_they, contractions_he, contractions_it, contractions_she,
                                       contractions_we, contractions_i, contractions_you]

    for pronoun in pronouns_that_take_contractions:
        for contraction in pronoun:
            if input[:len(contraction)] == contraction:
                if firstword(original_output) != GetPronounFromContraction(contraction):
                    return GetPronounFromContraction(contraction) + " " + uncapitalize(original_output)

    # Solution 2:  Append beginning of input
    if input.find(firstword(original_output)) < 15 and input.find(firstword(original_output)) > 0:
        return capitalize(input[:input.find(firstword(original_output))] + original_output)
    if input.find(firstword(original_output).lower()) < 15 and input.find(firstword(original_output).lower()) > 0:
        return capitalize(input[:input.find(firstword(original_output).lower())] + uncapitalize(original_output))

    # Solution 3 was axed

    # All good, return original
    return original_output
#endregion


#region API
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

import jwt
from passlib.hash import bcrypt
from tortoise import fields
from tortoise.contrib.fastapi import register_tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
from tortoise.models import Model
from pydantic import BaseModel, EmailStr # EmailStr is new

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Repsonse")
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import sqlite3
# Connecting to database
conn = sqlite3.connect('db.sqlite3')

# Creating a cursor object using the cursor() method
cursor = conn.cursor()


import hashlib


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


logger.debug(md5(encoder_path))
logger.debug(md5(decoder_path))
logger.debug(md5(init_decoder_path))


JWT_SECRET="GC795"


class User(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(50, unique=True)
    password_hash = fields.CharField(128)
    plan = fields.CharField(20)
    budget = fields.IntField(pk=False)
    active = fields.BooleanField()

    def verify_password(self, password):
        return bcrypt.verify(password, self.password_hash)


User_Pydantic = pydantic_model_creator(User, name='User')
UserIn_Pydantic = pydantic_model_creator(User, name='UserIn', exclude_readonly=True)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')


async def authenticate_user(username: str, password: str):
    user = await User.get(username=username)
    if not user:
        return False
    if not user.verify_password(password):
        return False
    return user


class Creds(BaseModel):
    username: str
    password: str


@app.post('/token')
async def generate_token(inp: Creds):
    user = await authenticate_user(inp.username, inp.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password'
        )

    user_obj = await User_Pydantic.from_tortoise_orm(user)

    token = jwt.encode(user_obj.dict(), JWT_SECRET)

    return {'access token': token, 'token_type': 'bearer'}

"""
@app.get('/', response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})
"""

@app.post('/users', response_model=User_Pydantic)
async def create_user(user: UserIn_Pydantic):
    user_obj = User(
        username=user.username,
        password_hash=bcrypt.hash(user.password_hash),
        plan=user.plan,
        budget=user.budget,
        active=user.active
    )
    await user_obj.save()
    return await User_Pydantic.from_tortoise_orm(user_obj)


async def get_current_user(token: str=Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user = await User.get(id=payload.get('id'))
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password'
        )

    return await User_Pydantic.from_tortoise_orm(user)


@app.get('/users/me', response_model=User_Pydantic)
async def get_user(user: User_Pydantic = Depends(get_current_user)):
    return user


def verify_token(req: Request):
    token = req.headers["Authorization"]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user = User.get(id=payload.get('id'))
        return True
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password'
        )


def get_id(req: Request):
    token = req.headers["Authorization"]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload.get('id')
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password'
        )


def get_budget(req: Request):
    token = req.headers["Authorization"]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload.get('budget')
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password'
        )


def get_active(req: Request):
    token = req.headers["Authorization"]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload.get('active')
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password'
        )


register_tortoise(
    app,
    db_url='sqlite://db.sqlite3',
    modules={'models': ['main']},
    generate_schemas=True,
    add_exception_handlers=True
)


class Msg(BaseModel):
    input: str


class SumReq(BaseModel):
    summary: str
    original: str


@app.get("/")
async def root():
    return {"output": "Hello World. Welcome to FastAPI!"}


@app.get("/path")
async def demo_get():
    return {"output": "This is /path endpoint, use a post request to transform the text to uppercase"}

"""
@app.post("/path")
async def demo_post(
        inp: Msg,
        authorized: bool = Depends(verify_token),
        user_id: str = Depends(get_id),
        active: bool = Depends(get_active),
        budget: int = Depends(get_budget)
):
    if authorized:
        if not active or budget < 1:
            output = "Account is inactive or out of credits"
            return {"output": output}
        cursor.execute(f"UPDATE user SET budget = budget - 1 WHERE id = {user_id}")
        conn.commit()
        cursor.execute(f"SELECT * FROM user WHERE id = {user_id}")
        logger.debug(cursor.fetchall())
        output = await PostProcess(inp.input, SimplifyGate(cleanup(inp.input)))
        logger.debug(inp.input)
        logger.debug(output)
        return {"output": output}
    else:
        output = "Invalid login"
        return {"output": output}"""


@app.post("/path")
async def demo_post(inp: Msg):
    output = await PostProcess(inp.input, SimplifyGate(cleanup(inp.input)))
    logger.debug(inp.input)
    logger.debug(output)
    return {"output": output}


@app.post("/summary")
async def getSummary(
        inp: SumReq,
        # authorized: bool = Depends(verify_token),
        authorized: bool = True,
        # user_id: str = Depends(get_id),
        user_id: str = "Bobby",
        # active: bool = Depends(get_active),
        active: bool = True,
        # budget: int = Depends(get_budget)
        budget: int = 9001
):
    if authorized:
        if not active or budget < 1:
            output = "Account is inactive or out of credits"
            return {
                "summary": output,
                "rouge": 0
            }
        request_details = {"summary": inp.summary, "original": inp.original}
        res = requests.post("http://summarizer.railway.internal:7519/path", json=request_details)
        response = json.loads(res.content)
        logger.debug(inp.summary)
        logger.debug(response["summary"])
        return {
            "summary": response["summary"],
            "rouge": response["rouge"]
        }
    else:
        output = "Invalid login"
        return {"summary": output, "rouge": 0}


@app.get("/path/{path_id}")
async def demo_get_path_id(path_id: int):
    return {"output": f"This is /path/{path_id} endpoint, use post request to retrieve result"}


@app.get("/database")
async def demo_get_database():
    monn = sqlite3.connect('bobert.sqlite3')
    with monn:
        conn.backup(monn)
    monn.close
    return FileResponse(path='bobert.sqlite3',media_type="application/octet-stream",filename="bobert.sqlite3")
#endregion
