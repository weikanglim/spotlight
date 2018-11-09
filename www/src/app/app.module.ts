import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { TextSearchComponent } from './text-search/text-search.component';
import { ClassifierComponent } from './classifier/classifier.component';
import { HomeComponent } from './home/home.component';
import { DataComponent } from './data/data.component';
import { DataAddComponent } from './data-add/data-add.component';
import { FileDropModule } from 'ngx-file-drop';

@NgModule({
  declarations: [
    AppComponent,
    TextSearchComponent,
    ClassifierComponent,
    HomeComponent,
    DataComponent,
    DataAddComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FileDropModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
