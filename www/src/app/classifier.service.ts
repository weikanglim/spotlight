import { Injectable } from '@angular/core';
import { serverUrl } from '../environments/environment'
import { HttpClient, HttpHeaders, HttpErrorResponse, HttpResponse } from '@angular/common/http';
import {  Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { Classifier } from './classifier';
import { BaseService } from './base.service';

@Injectable({
  providedIn: 'root'
})
export class ClassifierService extends BaseService {

  constructor(private http: HttpClient) { super() }

  getClassifiers() : Observable<Classifier[]> {
    return this.http.get<Classifier[]>(`${serverUrl}/classifiers`)
      .pipe(catchError(this.handleError));
  }

  train(classifier : Classifier, dataFile: File, modelName : string) : Observable<{}> {
    var formData = new FormData();
    formData.append("dataFile", dataFile);
    formData.append("modelName", modelName);
    
    console.log(`${serverUrl}/classifiers/${classifier.id}/train`);
    return this.http.post(`${serverUrl}/classifiers/${classifier.id}/train`, formData)
      .pipe(catchError(this.handleError));
  }
}
