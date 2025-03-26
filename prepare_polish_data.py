import json
import os

# This script prepares Polish data for REFT training
# We'll create a dataset of Polish instruction-response pairs

# Example Polish training data - replace with your own high-quality data
# Format: [instruction, response]
polish_examples = [
    ["Opisz historię Polski w kilku zdaniach.", "Polska to kraj o bogatej tysiącletniej historii. Powstała jako państwo w X wieku. Przez wieki przechodziła różne koleje losu, w tym okres rozbiorów (1795-1918), kiedy zniknęła z mapy Europy. Odzyskała niepodległość po I wojnie światowej, przetrwała II wojnę światową i okres komunizmu. Od 1989 roku jest demokratycznym państwem, a od 2004 członkiem Unii Europejskiej."],
    ["Jakie są najpopularniejsze dania kuchni polskiej?", "Do najpopularniejszych dań kuchni polskiej należą: bigos (danie z kapusty i mięsa), pierogi (ciasto z różnymi nadzieniami), żurek (zakwaszana zupa na bazie mąki żytniej), kotlet schabowy (panierowany kotlet wieprzowy), gołąbki (mięso z ryżem zawinięte w liście kapusty), placki ziemniaczane oraz sernik i szarlotka jako desery."],
    ["Wyjaśnij, dlaczego język polski jest uważany za trudny do nauczenia.", "Język polski jest uznawany za trudny z kilku powodów: posiada siedem przypadków gramatycznych, skomplikowaną odmianę czasowników, rozbudowany system rodzajów gramatycznych, trudną wymowę (szczególnie spółgłoski szumiące, ciszące i dwuznaki), oraz wiele wyjątków od reguł. Dodatkowo, akcent w języku polskim pada zwykle na przedostatnią sylabę, a pisownia zawiera znaki diakrytyczne (ą, ę, ć, ł, ń, ó, ś, ź, ż)."],
    ["Napisz krótki wiersz o wiośnie w Polsce.", "Wiosna w Polsce\n\nPrzebiśniegi nieśmiało wychylają główki,\nSłońce coraz śmielej przebija się przez chmurki.\nBociany wracają do swoich gniazd starych,\nA w parkach spacerowicze w kurtkach już nie szarych.\n\nPowietrze pachnie świeżością i nadzieją,\nPąki na drzewach w zieleń się odzieją.\nWiosna w Polsce to czas odrodzenia,\nGdy natura budzi się do życia z zimowego uśpienia."],
    ["Jak zapytać o drogę po polsku?", "Oto kilka przydatnych zwrotów, aby zapytać o drogę po polsku:\n- Przepraszam, jak dojść do...? (Excuse me, how do I get to...?)\n- Gdzie jest najbliższy przystanek autobusowy? (Where is the nearest bus stop?)\n- Czy może mi Pan/Pani powiedzieć, jak dotrzeć do centrum? (Could you tell me how to get to the center?)\n- Przepraszam, jestem zagubiony/zagubiona. (Excuse me, I'm lost.)\n- Czy to jest dobra droga do...? (Is this the right way to...?)"],
    ["Przetłumacz następujące zdanie na język angielski: 'W Warszawie jest wiele interesujących zabytków i muzeów.'", "In Warsaw, there are many interesting monuments and museums."],
    ["Wymień główne miasta Polski i opisz krótko każde z nich.", "Warszawa - stolica i największe miasto Polski, centrum polityczne, kulturalne i gospodarcze.\nKraków - dawna stolica z pięknym Starym Miastem i Wawelem, ważny ośrodek kultury i nauki.\nŁódź - centrum przemysłu włókienniczego z bogatą historią fabrykanckiej architektury.\nWrocław - miasto położone na 12 wyspach, znane z krasnali i bogatej architektury.\nPoznań - ważny ośrodek gospodarczy i akademicki, znany z targów międzynarodowych.\nGdańsk - główny port morski Polski, część Trójmiasta, miejsce narodzin Solidarności.\nSzczecin - port nad Odrą, ważny ośrodek komunikacyjny i przemysłowy."],
    ["Jakie są tradycyjne polskie święta i jak są obchodzone?", "Najważniejsze polskie święta to:\nWigilia i Boże Narodzenie (24-26 grudnia) - rodzinne święto z tradycjami jak dzielenie się opłatkiem, 12 potraw wigilijnych, kolędy.\nWielkanoc - najważniejsze święto religijne z tradycjami jak święcenie pokarmów, śmigus-dyngus.\nAndrzejki (29/30 listopada) - wieczór wróżb i zabaw.\nDzień Wszystkich Świętych (1 listopada) - dzień pamięci o zmarłych, odwiedzanie cmentarzy.\nDożynki - święto plonów obchodzone pod koniec lata/na początku jesieni.\nŚwięto Konstytucji 3 Maja i Święto Niepodległości (11 listopada) - święta państwowe."],
    ["Wyjaśnij różnicę między 'być' i 'mieć' w języku polskim i podaj przykłady.", "W języku polskim 'być' (to be) i 'mieć' (to have) to dwa podstawowe czasowniki o różnym znaczeniu i odmianie:\n\n'Być' wyraża istnienie, określa stan lub cechę:\n- Jestem Polakiem. (I am Polish.)\n- On jest wysoki. (He is tall.)\n- Byliśmy w Krakowie. (We were in Krakow.)\n\n'Mieć' wyraża posiadanie lub obowiązek:\n- Mam psa. (I have a dog.)\n- Oni mają duży dom. (They have a big house.)\n- Mieliśmy dużo pracy. (We had a lot of work.)\n- Mam zrobić zakupy. (I have to do shopping.)"],
    ["Jak poprosić o pomoc w sytuacji awaryjnej w Polsce?", "W sytuacji awaryjnej w Polsce możesz użyć następujących zwrotów:\n- Pomocy! (Help!)\n- Potrzebuję pomocy! (I need help!)\n- Proszę wezwać pogotowie/policję/straż pożarną! (Please call an ambulance/police/fire brigade!)\n- Miałem/miałam wypadek. (I had an accident.)\n- Zgubiłem/zgubiłam się. (I got lost.)\n\nNumery alarmowe:\n112 - ogólny numer alarmowy\n999 - pogotowie ratunkowe\n998 - straż pożarna\n997 - policja"]
]

# Create directory for data if it doesn't exist
if not os.path.exists("polish_data"):
    os.makedirs("polish_data")

# Save the examples in a format that can be used for REFT training
with open("polish_data/polish_examples.json", "w", encoding="utf-8") as f:
    json.dump(polish_examples, f, ensure_ascii=False, indent=2)

print("Polish training data prepared and saved to polish_data/polish_examples.json") 