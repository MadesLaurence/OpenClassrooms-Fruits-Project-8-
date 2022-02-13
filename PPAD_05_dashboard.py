import altair as alt
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from matplotlib import figure
import matplotlib.pyplot as plt


############### Déclaration des constantes #################################################### 
# Demande par défaut à la première connection
DEMANDEDEF = 0
# Taille du point de la demande
DEM_SIZE=200
# Taille des autres points
DOT_SIZE=10
# Couleurs
VERT = "#00A76C"
ORANGE = "#FDA501"
VERT_CLAIR = "#00EA96"
ORANGE_CLAIR = "#FEC358"    
VIOLET = "#7030A0"

PROBA_NAME = "Probabilité de défaut"

############### Déclaration des fonctions #################################################### 
@st.cache(allow_output_mutation=True)
def load_data(nrows):  
    
    df_desc = pd.read_csv("PPAD_04_dash_desc.zip", compression='zip', nrows=nrows)
    df_pred = pd.read_csv("PPAD_04_dash_pred.zip", compression='zip', nrows=nrows)
    df_imp = pd.read_csv("PPAD_04_feature.csv")
    df_cst = pd.read_csv("PPAD_04_cost.csv")
    return df_desc, df_pred, df_imp, df_cst

@st.cache(allow_output_mutation=True)
def graphe_jauge_score(df, threshold, line):
    
    threshold = np.round(threshold * 100, 1)
            
    # Représenter le fond
    data = pd.DataFrame([['', 'Refus', 100-threshold], ['', 'Accord', threshold]], 
                        columns=['X', 'Décision', PROBA_NAME])        
    bar = alt.Chart(data).mark_bar().encode(
                x=alt.X('X', title=''), y=alt.Y(PROBA_NAME), 
                order=alt.Order('Décision', sort='ascending'),
                color=alt.condition(alt.datum.Décision == 'Refus', alt.value(ORANGE), alt.value(VERT))
                                            ).properties(width=300, height= 500, title='')                               

    # Représenter le seuil
    seuil = alt.Chart(pd.DataFrame([['', threshold]], columns=['X', PROBA_NAME]))\
        .mark_tick(color='darkred', thickness=2, size=300*0.8).encode(x=alt.X('X'),y=alt.Y(PROBA_NAME))    
    text_seuil = seuil.mark_text(align='center', baseline='middle', dy=-10, color='darkred').encode(text=PROBA_NAME+':Q')
        
    # Représenter la demande    
    val = np.round(df[PROBA_NAME].iloc[line] * 100, 1)
    df_dem = pd.DataFrame([['', val]], columns=['X', PROBA_NAME])        
    demande = alt.Chart(df_dem).mark_tick(color='purple', thickness=2, size=300*0.8).encode(x=alt.X('X'), 
                                                                                            y=alt.Y(PROBA_NAME))    
    text_demande = demande.mark_text(align='center', baseline='middle', dy=-10, color='purple').encode(text=PROBA_NAME+':Q')
    
    return bar+seuil+text_seuil+demande+text_demande, val

@st.cache(allow_output_mutation=True)
def graphe_importance_locale(df_desc, df_norm, df_weight, line, top_nb):
    
    # Récupérer les valeurs normalisées du sample
    df_temp = pd.DataFrame(df_norm.iloc[line]) 
    df_temp.reset_index(inplace=True)
    df_temp.columns=("Feature", "Value")

    # Calculer l'importance locale des variables ( Weight * Value )
    df_res = df_weight.merge(df_temp, on='Feature')
    df_res['Importance'] = df_res['Weight']*df_res['Value']

    # Trier par ordre d'importance locale croissante
    df_res.sort_values('Importance', ascending=True, inplace=True)

    # Conserver les top_nb variables dans les 2 sens
    df_pos = df_res[df_res['Importance']<0].head(top_nb)
    df_neg = df_res[df_res['Importance']>0].tail(top_nb)
    df_res = pd.concat([df_pos, df_neg])
    
    # Récupérer les valeurs brute du sample
    df_temp = pd.DataFrame(df_desc.iloc[line]) 
    df_temp.reset_index(inplace=True)
    df_temp.columns=("Feature", "Valeur")   
    df_res = df_res.merge(df_temp, on='Feature')
    df_res['Valeurs_txt'] = np.round(df_res['Valeurs'],2).astype(str) 
    df_res['Valeur']= df_res['Valeurs_txt']
                
    c = alt.Chart(df_res).mark_bar().encode(x=alt.X('Importance', title='Impact'), 
                                            y=alt.Y('Code Variable:N', sort='-x', title='Variables'), 
                                            tooltip=['Description', 'Valeur', 'Code Variable'],           
                                            color=alt.condition(alt.datum.Importance > 0, alt.value(ORANGE), alt.value(VERT))
                                            ).properties(height=400, title='Impacts des variables sur la décision')
    
    return c
    
@st.cache(allow_output_mutation=True)
def graphe_aire_score_monovariable(df, x_var_name, x_var_pas, title, x_lab, mean_factor, line):    
    
    y_var_name = PROBA_NAME
    
    # Calculer la moyenne par var_name
    gb = df.groupby([x_var_name])
    df_data = pd.DataFrame(gb[y_var_name].mean())
    df_data[y_var_name] = df_data[y_var_name] * 100
    df_data['Top'] = 100    
    df_data.reset_index(drop=False, inplace=True)
            
    # Aire refus en orange
    area1 = alt.Chart(df_data).mark_area(color=ORANGE).encode(
        alt.X(x_var_name, scale=alt.Scale(zero=False, nice=False), title=x_lab),
        alt.Y('Top', title=y_var_name),
        opacity=alt.value(0.8))
    
    # Aire accord en vert
    area2 = alt.Chart(df_data).mark_area(color=VERT).encode(
        alt.X(x_var_name, scale=alt.Scale(zero=False, nice=False), title=x_lab),
        alt.Y(y_var_name, title=y_var_name),
        opacity=alt.value(0.8))
    
    # Représenter la demande  
    df_dem = df[df.index == line][[x_var_name, y_var_name]]
    demande = alt.Chart(df_dem).mark_circle(size=DEM_SIZE).encode(x=alt.X(x_var_name), y=alt.Y(y_var_name), color=alt.value(VIOLET)).properties(title=title)
    
    return area1+area2+demande
    
@st.cache(allow_output_mutation=True)
def graphe_scatter_decision_bivariables(df, x_var_name, y_var_name, title, x_lab, y_lab, line):

    # Représenter le nuage de points 
    nuage = alt.Chart(df).mark_circle(size=DOT_SIZE).encode(x=alt.X(x_var_name, title=x_lab), y=alt.Y(y_var_name, title=y_lab), 
        color=alt.condition(alt.datum.PRED == 1, alt.value(ORANGE), alt.value(VERT)))
    
    # Représenter la demande  
    df_dem = df[df.index == line][[x_var_name, y_var_name]]
    demande = alt.Chart(df_dem).mark_circle(size=DEM_SIZE).encode(x=alt.X(x_var_name), y=alt.Y(y_var_name), color=alt.value(VIOLET)).properties(title=title)

    return nuage+demande

#@st.cache(allow_output_mutation=True)
#def graphe_fonction_cout(df, threshold, line):
#    plot = plt.figure(figsize=(12, 8))
#    plt.plot(df['Seuil']*100, df['Cout'])
#    plt.xlabel("Seuil de probabilité")
#    plt.ylabel("Coût")
#    plt.title("Coût en fonction du seuil de probabilité")
#    cost_min = 0; cost_max = df['Cout'].max(); # Calculer les coordonnées du minimum
#    plt.plot([threshold*100, threshold*100], [cost_min + 1, cost_max], color='darkred')    
#    label = "Seuil : "+str(np.round(threshold*100, 1))+"%, Coût global : "+str(np.round(df['Cout'].min(), 1))+"M€"
#    plt.annotate(label, (threshold*100+2, 16), textcoords="offset points", xytext=(0, 0), ha='left', color='darkred') 
#    plt.plot([proba_dem, proba_dem], [cost_min, cost_max], color='purple')   
#    label2 = "Demande : "+str(proba_dem)+"%"
#    plt.annotate(label2, (proba_dem+2, 0), textcoords="offset points", xytext=(0, 0), ha='left', color='purple') 
#    return plot

def lire_ligne():
    file=open("PPAD_05_ligne.txt", "r")
    line=file.read()
    file.close()
    return line

def ecrire_ligne(line):
    file=open("PPAD_05_ligne.txt", "w+")
    file.write(str(line))
    file.close()
    
    
############### Préparation #################################################### 
# Imposer une apparence en mode large
st.set_page_config(page_title="Explication de la décision", initial_sidebar_state="expanded", layout="wide")

# Charger les données
df_dash_desc, df_dash_pred, df_feature_imp, df_cost = load_data(500)

# Pour renommer les variables
df_ref = pd.DataFrame({
    'Feature':[     'AMT_GOODS_PRICE',       
                    'EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3',
                    'DAYS_EMPLOYED',
                    'DAYS_EMPLOYED_PERCENT',        
                    'CREDIT_TERM',         
                    'EXT_SOURCE_2 EXT_SOURCE_3^2',
                    'client_installments_AMT_PAYMENT_min_sum',
                    'DAYS_BIRTH_x',
                    'AMT_CREDIT',
                    'bureau_DAYS_CREDIT_max'],                 
    'Code Variable':[   'AMT_GOODS_PRICE',       
                    'EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3',
                    'YEARS_EMPLOYED',
                    'DAYS_EMPLOYED_PERCENT',        
                    'CREDIT_TERM',         
                    'EXT_SOURCE_2 EXT_SOURCE_3^2',
                    'AMT_PAYMENT_min_sum_K',
                    'YEARS_BIRTH',
                    'AMT_CREDIT_K',
                    'bureau_YEARS_CREDIT_max'],
    'Description':['Prix du bien concerné par la demande',                   
                    'Combinaison linéaire Scores externes 1, 2, 3',
                    'Ancienneté sur le poste actuel (années)',                     
                    'Ratio Ancienneté sur le poste courant et Age du client',
                    'Ratio Annuité demandé et Montant demandé',                     
                    'Combinaison linéaire Scores externes 2 et 3 au carré',
                    'Somme des montants de la plus petite échéance payée sur chacun des prêts antérieurs contracté chez "Prêt à dépenser" (K€)',
                    'Age du client (années)',                     
                    'Montant demandé (K€)',                
                    'Ancienneté du plus ancien des crédits reportés par le Crédit Bureau']})

# Récupérer seuil à partir de df_cost
threshold_min = df_cost['Seuil'].iloc[df_cost['Cout'].idxmin()]

# Préparation des variables pour affichage plus signifiant pour le métier
df_dash_desc['YEARS_BIRTH'] = np.abs(df_dash_desc['DAYS_BIRTH_x'] / 365)
df_dash_desc['YEARS_EMPLOYED'] = - df_dash_desc['DAYS_EMPLOYED'] / 365
df_dash_desc['AMT_CREDIT_K'] = df_dash_desc['AMT_CREDIT'] / 1000
df_dash_desc['bureau_YEARS_CREDIT_max'] = df_dash_desc['bureau_DAYS_CREDIT_max'] / 365
df_dash_desc['AMT_PAYMENT_min_sum_K'] = df_dash_desc['client_installments_AMT_PAYMENT_min_sum'] / 1000

ligne_txt = lire_ligne()
if ligne_txt == "":    ligne_txt = DEMANDEDEF
ligne=int(ligne_txt)

##################################################################################### 
############### Bandeau vertical #################################################### 
##################################################################################### 
with st.sidebar:
        
    # Bouton de tirage aléatoire de ligne        
    st.sidebar.title('Choisissez votre demande')
    if st.sidebar.button('Tirer une demande aléatoirement'): 
        ligne = np.random.randint(0, high=len(df_dash_desc), size=1)[0] 
        ecrire_ligne(ligne)

    # Menu 
    selected = option_menu("Choisissez votre thème", ['Explication de la décision',
                                                   'Le client', 
                                                   'La demande de prêt', 
                                                   'Les anciens prêts', 
                                                   'La demande représentée sur la fonction de coût'], 
                                                   icons=['dot', 'dot', 'dot', 'dot', 'dot'], menu_icon="cast", default_index=0)

    st.sidebar.write('______')

    # Afficher la décision -----------------------------
    bg_col=VERT; col=VERT_CLAIR; dec= 'Accord';   
    if df_dash_pred['PRED'].iloc[ligne] == 1: 
        bg_col=ORANGE; col=ORANGE_CLAIR; dec= 'Refus';
    style='"background-color:'+bg_col+';color:'+col+';text-align:center;font-size:24px"'
    html=f'<h1 style='+style+'>'+dec+'</h1>'
    st.sidebar.markdown(html, unsafe_allow_html=True)

    st.sidebar.write('______')

    # Afficher la jauge  -----------------------------
    j, proba_dem = graphe_jauge_score(df_dash_desc, threshold_min, ligne)        
    st.sidebar.altair_chart(j, use_container_width=True) 
        

##################################################################################### 
############### Espace principal #################################################### 
##################################################################################### 
# Afficher la demande et la décision
st.title('Demande ' + str(ligne)+' > '+dec)
st.write('______')

if selected == "Explication de la décision":
    # Représenter l'importance des variables pour la demande -----------------------------
    st.subheader(selected+' : Impact des variables sur la probabilité de défaut')

    comment1 = '<p style="font-family:sans-serif; color:Brown; font-size: 15px"> - Un impact positif signifie que la variable fait accroître la probabilité de défaut.</p>'
    comment2 = '<p style="font-family:sans-serif; color:Darkgreen; font-size: 15px"> - Un impact négatif signifie que la variable fait décroître la probabilité de défaut.</p>'
    st.markdown(comment1+comment2, unsafe_allow_html=True)    
    df_feature_imp_enriched = pd.merge(df_feature_imp, df_ref, on='Feature') 
    
    df_dem = df_dash_desc[df_dash_desc.index == ligne].T
    df_dem.rename(columns = {df_dem.columns[0]:'Valeurs'}, inplace=True)
    df_dem.reset_index(drop=False, inplace=True) 
    df_dem.rename(columns={'index':'Code Variable'}, inplace=True)
    df_feature_imp_enriched = pd.merge(df_feature_imp_enriched, df_dem, on='Code Variable') 
        
    i = graphe_importance_locale(df_dash_desc, df_dash_pred, df_feature_imp_enriched, ligne, 5)
    st.altair_chart(i, use_container_width=True) 

    # Présenter la tables des variables décrites et valorisées -----------------------------
    st.subheader('Les informations du client')   
    st.markdown('- Triées par importance globale des variables')                      
    st.table(df_feature_imp_enriched[['Code Variable', 'Description', 'Valeurs']])

############### Le client #################################################### 
if selected == "Le client":   
    st.subheader(selected)    

    with st.container():
        gauche, droite = st.columns(2)
        with gauche:
            c = graphe_aire_score_monovariable(df_dash_desc, 'EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3', 0.05, \
                                           "Impact des scores externes", "a x Score1 + b x Score2 + c x Score3", 10000, ligne)
            st.altair_chart(c, use_container_width=True) 
        with droite:
            c = graphe_aire_score_monovariable(df_dash_desc, 'EXT_SOURCE_2 EXT_SOURCE_3^2', 0.05, \
                                           "Impact des scores externes", "d x Score2 + e x Score3^2", 10000, ligne)
            st.altair_chart(c, use_container_width=True) 

    with st.container():
        gauche, droite = st.columns(2)
        with gauche:
            c = graphe_aire_score_monovariable(df_dash_desc, 'DAYS_EMPLOYED_PERCENT', 0.05, \
                                "Impact du ratio Ancienneté / Age", "Ratio Ancienneté / Age", 10000, ligne)
            st.altair_chart(c, use_container_width=True)     
        with droite:

            c = graphe_scatter_decision_bivariables(df_dash_desc, 'YEARS_BIRTH', 'YEARS_EMPLOYED', \
                                                'Répartitions des décisions', \
                                                'Age du client (années)', 'Ancienneté sur le poste actuel (années)', ligne)
            st.altair_chart(c, use_container_width=True) 

############### La demande de prêt #################################################### 
if selected == "La demande de prêt":   
    st.subheader(selected)

    with st.container():
        gauche, droite = st.columns(2)
        with gauche:
            c = graphe_aire_score_monovariable(df_dash_desc, 'AMT_GOODS_PRICE', 100, \
                                                   "Impact du prix du bien", "Prix du bien", 10000, ligne)
            st.altair_chart(c, use_container_width=True) 

    with st.container():
        gauche, droite = st.columns(2)
        with gauche:
            c = graphe_aire_score_monovariable(df_dash_desc, 'CREDIT_TERM', 0.05, \
                                "Impact du ratio Annuité / Montant", "Ratio Annuité / Montant", 10000, ligne)
            st.altair_chart(c, use_container_width=True)
        with droite:
            c = graphe_aire_score_monovariable(df_dash_desc, 'AMT_CREDIT', 0.05, \
                                "Impact du Montant du prêt demandé", "Montant du prêt", 10000, ligne)
            st.altair_chart(c, use_container_width=True)

############### Les anciens prêts #################################################### 
if selected == "Les anciens prêts":
    st.subheader(selected)

    st.markdown('"Crédit Bureau"')    
    c = graphe_aire_score_monovariable(df_dash_desc, 'bureau_YEARS_CREDIT_max', 10, \
                                   "Impact de l'ancienneté du plus ancien des crédits reportés par le Crédit Bureau",\
                                   "Ancienneté la plus grande (en années)", 100, ligne)
    st.altair_chart(c, use_container_width=True)

    st.markdown('"Prêt à dépenser"')
    c = graphe_aire_score_monovariable(df_dash_desc, 'AMT_PAYMENT_min_sum_K', 10, \
                                       "Impact des échéances payées des prêts antérieurs",\
                                       "Somme des montants de la plus petite échéance payée sur chacun des prêts antérieurs", 100, ligne)
    st.altair_chart(c, use_container_width=True)

############### La demande représentée sur la fonction de coût #################################################### 
if selected == "La demande représentée sur la fonction de coût":    
    st.subheader(selected)
    
    # Afficher la fonction de coût -----------------------------
#    st.pyplot(graphe_fonction_cout(df_cost, threshold_min, ligne))
