def filter_data_with_job_title_oo(classlist,jobtitle_name,index):
    result_list = []
    if index == 1:
        for i in range(len(classlist)):
            if classlist[i].title.find(jobtitle_name)!=-1:
                result_list.append(classlist[i])
    if index == 2:
        for i in range(len(classlist)):
            if classlist[i].title.find(jobtitle_name)==-1:
                result_list.append(classlist[i])
    return result_list