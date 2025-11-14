from typing_extensions import TypedDict
from typing import Annotated,List,Literal
import operator
from pydantic import BaseModel,Field

class Section(BaseModel):
    title:str=Field(description="Title of the section")
    description:str = Field(description="Description about the section based on the title given")
    
class Sections(BaseModel):
    sections:List[Section] = Field(description="A list of sections in the report")
    
class UserInput(TypedDict):
    title:str 
    about_problem:str 
    methods_used:str
    proposed_workflow:str 
    results:str

class AutoState(TypedDict):
    topic: str
    sections: list[Section] # default empty list   # default empty list
    final_report: str 
    
class UserState(TypedDict):
    user_input:UserInput
    abstract:str
    intro:str 
    methodology:str 
    proposed_method:str 
    results:str
    references:str 
    conclusion:str
     
class State(TypedDict):
    user: UserState 
    auto: AutoState  
    is_userInput:Literal[True,False]
    completed_sections:Annotated[list,operator.add]
    
class WorkerState(TypedDict):
    section:Section
    completed_sections:Annotated[list,operator.add]