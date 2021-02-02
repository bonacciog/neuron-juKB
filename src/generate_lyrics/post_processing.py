import re


def postprocess(text, original_text ):

    # replace <nl> with \n
    text = re.sub(r"<nl>", r"\n", text)

    punct_to_remove = ".;!:°#§+*][^|\\£$%&/=><_-"
    for punct in punct_to_remove:
        text = text.replace(punct,"")
    
    text = text.replace(u"\ufffd", "")
    
    # replace \n with spaces with only spaces
    text = re.sub(' +', ' ', text)

    # remove capital letters after \n
    text = '\n'.join( map( lambda s: s.strip().capitalize(), text.split('\n') ) )
    
    if text.startswith(original_text.capitalize() + "\n"):
        text = '\n'.join( text.split('\n')[1:]).strip() 
    

    last_line = text.split("\n")[-1]
    
    if len(last_line.split(' ')) < 4:
        text = '\n'.join(text.split("\n")[:-1]).strip()


    return text