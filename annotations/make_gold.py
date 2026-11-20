import pandas as pd, sys
SEG = """boʻy-i-cha yangi amal-iy osh-ir-ish haq-i-da et-il-gan son umum-iy boʻl-ish-i xalq-aro kel-ib inson xalq yaʼni zamon-aviy yana taʼsir biz nima jihat-dan
teg-ish-li yordam-i-da samara-li kichik gʻa aniq nazorat ona kel-tir-il-gan oʻsha tuz-il-ish-i ayrim kun-da yer-da mahalliy sport qoʻl-la-sh oʻz-i-ni xulosa buyuk
tur-lar-i texnologik yuza-ga boʻl-ish kam oʻzbekiston-da sinf et-ib koʻr-sat-il-gan tartib-ga modda kishi murakkab ich-i-da bit-ta nashr bu-ning navoiy yarat-il-gan
soʻz-lar odam-lar xususiyat-lar-i-ni tavsiya-lar tahlil-i ortiq qayd sabab-li demak top-ing ish-chi hali zarurat-i dastur-i oʻt-ish et-di yangi-lig-i yil-lik bozor
kimyo kadr-lar erkin masala-lar et-il-ish-i islom qon tur-ib maqsad-ga aloqa mashhur rasm-iy prezident-i adabiyot ket-gan foyda-lan-ib obyekt-i ilgari the vatan
tarkib-iy sharq boy yuqori-da avval kel-adigan hozir boj-xona yoz-ma dars-lik yoʻl-ga bugun qol-adi biz-ga tizim-li boʻy-la-b tuz-ish omil-lar nom-la-n-gan his
qil-ish-ni strategiya-si oʻrgan-il-gan boʻl-in-adi singari fizika doim qatʼiy oʻrgan-ib keng-ay-tir-ish aka maʼrifat old-in-gi amir kuch-i-ni koʻr-gan ifoda-lo-vchi
ona-m mis ixtiyor-iy hajm-i-ni shamol aylan-adi resurs yorqin kishi-ning kut-ib rahmat qil-di-m qil-moqda modda-ning ijod-kor namuna ek-ish taolo yil-i-ga kel-ish
bari-bir dar-haqiqat aylan-ish koʻz-dan davr-i-ga kun-lik oy-lik aylan-ma taʼrif-i uskuna-lar-i aʼzo-lar-i-ning shart-i google maktab-i-ning samara-dor-lik
soha-si-ning vazifa-lar-i-dan kecha-si lozim-ki qimor navroʻz material-lar-dan fonetik kitob-xon buyrak band-lig-i-ni ustuvor-lig-i bor-ish-ga minut shakl-la-n-adi
hujjat-lar-ning boʻl-ish-i-ning muvaffaqiyat och-il-ib katta-roq klaster chet-da mamlakat-ni qop-la-sh hozir-cha qayer-ga imzo hazrat-i paygʻambar-imiz yur-adi
tajriba-lar-i-ni sintez-i epik sev-ib boshoq-li foiz-ni qavat tag-i-da oʻrgan-ish-dan tafakkur-ning erta-ngi kes-ish qil-in-sa mamlakat-da-gi masala-da tovuq
hamshira-lik men-im-cha xarakter-i-ni formula-si-ni dadil yech-ish-ga parvoz vaqt-lar-i-ni qoyil zulfiya kredit-lar taʼzim inflyatsiya oʻt-kaz-il-gan-da robinzon
yer-lar-ga yot-ar-di yurt-ga bola-kay soʻr-ov-lar konstruksiya-si-ni qoʻl-la-n-il-ayotgan qon-li yasa-l-ish-i yoq sil tur-il-adi zarra-cha-lar-i okean xon-lig-i
atsetilen osh-ma-s-lig-i de-moqchi intensiv-lik nafas-i nogʻora sovet-lar tayan-ish energiya-si-dan kutubxona-lar sino-ning qoʻl-la-n-ib kurt kitob-xon-ni bosh-pana
konserva-la-sh shunday-mi vrach-lik tarbiya-si-da-gi aksiya-lar-i dehqon-chilik-ning qofiya ajrat-ing fazilat-lar-ga universitet-lar be-tob toʻlqin-lar-i-ning
toʻy-ga prinsip-lar-ga chuqur-lash-tir-il-gan yori-sh-ib dada-jon kovak tomon-lar-i-dan tut-sin ayt-ish-imiz nuarte perifraza qisqar-tir-ib oʻra-l-ib
qol-diq-lar-i-ni yarat-ayotgan kir-im muhandis-lar-ni quddus aks-lan-tir-ish antipiren burchag-i-ni farzand-li ogʻ-ish-lar boʻl-ma-s-mi-di muz-la-t-il-gan sulh
bosqin-chi-lar-i maʼno-lar-i-ga saltanat-i-ning tez-lash-ib zahira-lar-i modda-lar-dir odat-lar-imiz singarmonizm aylan-ish-i-ning chalqancha mikro-element-lar-i
shosh-may yuksal-adi zulxumor dastgoh-lar-i-ning his-lar-i-ni koʻr-in-uvchi koʻr-sat-il-gan-lig-i mixaylov bajar-moqda chuvalchang erkin-lik-lar-ga konvert-ning
maktub-i nagoya tish-lar-ning assotsiativ-lik farzand-lar-i-dan gunoh-siz gʻuncha-oy kir-ish-ar super-liga-si tuna-b vakant daraxt-i-dan molyar-li novda-lar-i-da
oʻyna-gan-i zabur zaratushtra asr-imiz boʻl-ish-sin eri-ma-gan faktor-lar-ga fariduddin joy-lash-uv-i-ning lampa-lar matritsa-lar-i mentha mujassam-lash-tir-ish
notiq-lar och-koʻz-lik oz-i oʻt-mish-imiz-ni pufak-cha-lar rafiqov sing-gan tekis-lik-lar-dan uzat-di-m yigʻ-gan aralash-tir-ing basir chekka-si-dan
eʼtiqod-lar-i-ni gʻazab-lan-ib hisob-la-sh-lar-ning ikki-si-ga ilm-dir inflyatsiya-ga millat-imiz-ni oid-lig-i oson-dir pleonazm-ning rezervuar sehr-i-ga
suverenitet-i-ni vatar-lar vujud-i-da-gi vyetnam agʻana-b ajin-lar-i aralash-ma-si-da-gi builder diplom-ga etimologiya gilam-ni hosil-i-ga ish-la-ma-ydigan itur
koʻnik-ish mary savdo-lar-ni skaner-dan tashkilot-i-da-gi vee yech-im-lar-i-ning algebra-ga ansys asir-ni ato-vchi ber-ish-ma-gan chiq-ar-ish-adi
deklaratsiya-lar-i eftal-lar-ning fasohat-li holat-li ibodat-xona-lar-ga kripto-valyuta-lar-ga ovqat-lan-adi-lar oʻyla-yapti tekis-lo-vchi tiq-qan tuga-t-a-siz
turkman-lar almamatova bek-ni bit-moq bogʻ-la-n-ish-dir buvi-ng-ni dalil-ni epizod-lar-i eshit-ma-y-miz et-ish-mov-chi-lig-i-da koʻnik-ma-si-ning medalyon-ning
mukarrama-ga nozik-lik-lar-i qol-ma-y-siz raqs-i-ni sana-ma-lar shaboda shuhrat-i-ga soʻz-imiz-ni stafilokokk-lar suspenziya-lar tajriba-m-dan tur-ma-s-dan tut-il-a
aleyron belostok be-pisand-lik bloch boj-lar deyvid esirgapov faryod-im gemostaz hanafiy-lik iljay-ish-ib impuls-lar-ga iqtibos-iy janob-ga janon-ga jon-zot-lar-ga
kel-in-ayotgan-i-dan ket-lik-lar-da-gi kim-miz koʻpay-a-ver-di magnit-lo-vchi maradjabov mehriniso-ga modernization monax-lik oryol oxus ozigʻ-i-dir
oʻlcha-gich-ning oʻqi-gan-mi-siz polisiklik pomir-da qadam-da-gi qista-y-ver-gach qistosul qop-la-n-gan-mish quyruq-qa rajabov-lar risola-lar-i-dan rkp russell-ga
ruxsat-i-ni sanoat-lar-i-da shahar-i-ning sinqil-la-b siyrat-i-ga tabobat-ning tayyor-la-r-di-k taʼsir-lash-adigan tilsim-li toʻla-y-miz tugʻ-lar-i-ni tujamunit
urgʻochi-lar-i-ning usuliyat-i voqe-lan-gan-lig-i xaybar-ga xotira-lar-ingiz-ni xrizotil yigʻla-gan-i yosh-da-siz zarqaviy ziyorat-lar-i-ga""".split()
LOAN = set("""sport texnologik prezidenti obyekti kadrlar strategiyasi fizika resurs materiallardan fonetik minut klaster sintezi epik xarakterini formulasini
kreditlar inflyatsiya okean atsetilen intensivlik sovetlar energiyasidan konservalash vrachlik aksiyalari universitetlar prinsiplarga perifraza antipiren
singarmonizm mikroelementlari konvertning assotsiativlik superligasi vakant molyarli faktorlarga lampalar matritsalari inflyatsiyaga pleonazmning rezervuar
suverenitetini diplomga etimologiya skanerdan algebraga deklaratsiyalari epizodlari medalyonning stafilokokklar suspenziyalar aleyron gemostaz impulslarga
kriptovalyutalarga magnitlovchi monaxlik polisiklik xrizotil konstruksiyasini""".split())
PROPER = set("""oʻzbekistonda navoiy zulfiya robinzon sinoning quddus zulxumor mixaylov nagoya gʻunchaoy zaratushtra fariduddin rafiqov basir vyetnam almamatova
belostok deyvid esirgapov eftallarning maradjabov mehrinisoga oryol oxus pomirda rajabovlar russellga tujamunit xaybarga zarqaviy mukarramaga zabur""".split())
FOREIGN = set("the google mentha builder mary ansys bloch modernization".split())
OTHER = {"gʻa": "fragment, probably a split word", "itur": "fragment / unknown form", "vee": "unknown form, possibly OCR or foreign", "rkp": "abbreviation",
         "nuarte": "unknown form, possibly a name", "qistosul": "unknown form", "ozi": "ambiguous: oz+i ('a little of it') or misspelled oʻzi"}
NOTES = {"boʻyicha": "lexicalized postposition; historically boʻy+i+cha", "haqida": "lexicalized postposition; historically haq+i+da",
         "boʻylab": "lexicalized postposition; historically boʻy+la+b", "bitta": "bir+ta with assimilation r→t; surface split bit-ta",
         "ifodalovchi": "participle -(u)vchi; stem-final a→o (ifodala→ifodalo)", "atovchi": "stem ata- with a→o before -vchi",
         "tekislovchi": "tekisla- with a→o before -vchi", "magnitlovchi": "magnitla- with a→o before -vchi",
         "oldingi": "old+in+gi; -ingi treated as -in+gi", "burchagini": "k→g alternation (burchak→burchag)",
         "qisqartirib": "qisqar (become short) + tir + ib", "etishmovchiligida": "-mov- unproductive suffix (etishmov 'shortage')",
         "ketliklardagi": "probably from hyphenated ketma-ketliklardagi split at the hyphen", "darhaqiqat": "Persian prefix dar-",
         "mahalliy": "-iy not split: base mahalla is truncated", "qatʼiy": "-iy not split: base qatʼ is not an independent Uzbek word",
         "texnologik": "loan adjective kept whole (Russian -ik)", "polisiklik": "loan adjective kept whole",
         "chalqancha": "kept whole: chalqan- is not an independent stem", "tayyorlardik": "tayyorla+r+di+k (past habitual)",
         "oʻzbekistonda": "-iston not split (part of the toponym)", "erkin": "lexicalized; historically erk+in",
         "aylanadi": "aylan- treated as stem (lexicalized ayla+n)", "aralashtiring": "aralash- treated as stem"}
d = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
assert len(d) == len(SEG) == 500, (len(d), len(SEG))
bad = [(w, s) for w, s in zip(d.word, SEG) if s.replace("-", "") != w]
assert not bad, bad
d["segmentation"] = SEG
d["word_class"] = ["foreign" if w in FOREIGN else "proper" if w in PROPER else "other" if w in OTHER else "loan" if w in LOAN else "native" for w in d.word]
assert LOAN | PROPER | FOREIGN | set(OTHER) <= set(d.word), (LOAN | PROPER | FOREIGN | set(OTHER)) - set(d.word)
d["note"] = [OTHER.get(w, NOTES.get(w, "")) for w in d.word]
d["n_morphs"] = [s.count("-") + 1 for s in SEG]
d["annotator"] = "LLM annotation (Claude) — pending review by the authors"
d.to_csv("morph_gold_v1.csv", index=False, encoding="utf-8-sig")
print(d.word_class.value_counts().to_dict())
print(d.groupby("band").n_morphs.agg(["mean", "max"]).round(2))
print("words with ≥2 morphs:", int((d.n_morphs >= 2).sum()), "| boundaries total:", int((d.n_morphs - 1).sum()))
print("notes:", int((d.note != "").sum()))
