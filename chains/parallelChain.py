from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

first_model = ChatOpenAI()
second_model = ChatOpenAI()

first_prompt = ChatPromptTemplate.from_template(
    "Generate short and simple notes from the following text \n {text}"
)

second_prompt = ChatPromptTemplate.from_template(
    "Generate 5 short question answers from the following text \n {text}"
)

third_prompt = ChatPromptTemplate.from_template(
    "Merge the notes and quiz into a signle document \n notes -> {notes} quiz -> {quiz}"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": first_prompt | first_model | parser,
        "quiz": second_prompt | second_model | parser,
    }
)

merge_chain = third_prompt | first_model | parser

final_chain = parallel_chain | merge_chain

text = """
        Kernel ridge regression (KRR) [M2012] combines Ridge regression and classification (linear least squares with 
        -norm regularization) with the kernel trick. It thus learns a linear function in the space induced by the respective kernel and the data. For non-linear kernels, this corresponds to a non-linear function in the original space.

        The form of the model learned by KernelRidge is identical to support vector regression (SVR). However, different loss functions are used: KRR uses squared error loss while support vector regression uses 
        -insensitive loss, both combined with 
        regularization. In contrast to SVR, fitting KernelRidge can be done in closed-form and is typically faster for medium-sized datasets. On the other hand, the learned model is non-sparse and thus slower than SVR, which learns a sparse model for 
        , at prediction-time.

        The following figure compares KernelRidge and SVR on an artificial dataset, which consists of a sinusoidal target function and strong noise added to every fifth datapoint. The learned model of KernelRidge and SVR is plotted, where both complexity/regularization and bandwidth of the RBF kernel have been optimized using grid-search. The learned functions are very similar; however, fitting KernelRidge is approximately seven times faster than fitting SVR (both with grid-search). However, prediction of 100,000 target values is more than three times faster with SVR since it has learned a sparse model using only approximately 1/3 of the 100 training datapoints as support vectors.


"""

result = final_chain.invoke({"text": text})

print(result)
